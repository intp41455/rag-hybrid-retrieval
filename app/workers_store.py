"""Cloudflare Workers 适配层 —— 让同一套 Python 代码能在 R2/D1/KV 上运行。

架构说明：
  - 数据持久化：R2 bucket（entries JSON）+ KV（metadata + embeddings）
  - 嵌入式向量：调用 Agnes AI Embedding API（无需本地 BGE 模型）
  - 音视频转写：跳过（返回提示），或改为异步 Cloudflare Queue
  - 模型文件：不依赖本地 ONNX/PyTorch，全部走 API
"""
import json
import os
import uuid
import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Optional
import httpx

from app.config import settings
from app.models import Entry, Line, Chunk, Analysis, Visualization, Todo, Job, KnowledgeNode, KnowledgeEdge


# ────────────────────────────────────────────────────────────
# R2 / KV 适配器（替代本地文件系统）
# ────────────────────────────────────────────────────────────

_R2_BUCKET = os.getenv("R2_BUCKET_NAME", "rag-knowledge")
_KV_NS_ENTRIES = os.getenv("KV_NS_ENTRIES", "entries-meta")
_KV_NS_EMBED = os.getenv("KV_NS_EMBED", "embeddings")


def _r2_get(key: str) -> bytes | None:
    """从 R2 读取对象（Workers 环境通过 env 访问 R2；本地降级到文件系统）"""
    # Cloudflare Workers 环境
    try:
        import cloudflare  # type: ignore
        r2 = cloudflare.R2(env={"account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
                                "api_token":   os.getenv("CLOUDFLARE_API_TOKEN",   "")})
        obj = r2.bucket(_R2_BUCKET).get(key)
        return obj.bytes() if obj else None
    except Exception:
        pass
    # 本地降级：用文件系统
    path = os.path.join(settings.data_dir, "r2", key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


def _r2_put(key: str, data: bytes) -> None:
    try:
        import cloudflare
        r2 = cloudflare.R2(env={"account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
                                "api_token":   os.getenv("CLOUDFLARE_API_TOKEN",   "")})
        r2.bucket(_R2_BUCKET).put(key, data)
    except Exception:
        path = os.path.join(settings.data_dir, "r2", key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def _kv_get(ns: str, key: str) -> str | None:
    try:
        import cloudflare
        kv = cloudflare.WorkerKV(env={"account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
                                      "api_token":   os.getenv("CLOUDFLARE_API_TOKEN",   "")})
        return kv.get(ns, key)
    except Exception:
        path = os.path.join(settings.data_dir, "kv", ns, key)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        return None


def _kv_put(ns: str, key: str, value: str) -> None:
    try:
        import cloudflare
        kv = cloudflare.WorkerKV(env={"account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
                                      "api_token":   os.getenv("CLOUDFLARE_API_TOKEN",   "")})
        kv.put(ns, key, value)
    except Exception:
        path = os.path.join(settings.data_dir, "kv", ns)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, key), "w") as f:
            f.write(value)


# ────────────────────────────────────────────────────────────
# 替代本地 BGE：调用 Agnes AI Embedding API
# ────────────────────────────────────────────────────────────

_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 Agnes AI Embedding 接口，返回 embedding 向量列表。"""
    if not texts:
        return []
    api_key = settings.agnes_api_key
    if not api_key:
        # 无 API Key 时返回零向量（维度 128，兼容 BGE-small）
        return [[0.0] * 128 for _ in texts]
    base = "https://api.agnes-ai.cn/v1"
    url = f"{base}/embeddings"
    payload = {"model": _EMBED_MODEL, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        return [item["embedding"] for item in items]
    except Exception as e:
        print(f"[embed] API 失败，降级零向量: {e}")
        return [[0.0] * 128 for _ in texts]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ────────────────────────────────────────────────────────────
# Entry Store（替代本地 sqlite/文件）
# ────────────────────────────────────────────────────────────

class EntryStore:
    def __init__(self, data_dir: str | None = None):
        self.prefix = "entries/"

    def list_entries(self) -> list[dict]:
        raw = _r2_get("index.json")
        if not raw:
            return []
        index = json.loads(raw)
        entries = []
        for eid in index:
            blob = _r2_get(f"{self.prefix}{eid}.json")
            if blob:
                try:
                    entries.append(json.loads(blob))
                except Exception:
                    pass
        return entries

    def get_entry(self, entry_id: str) -> dict | None:
        blob = _r2_get(f"{self.prefix}{entry_id}.json")
        if not blob:
            return None
        return json.loads(blob)

    def save_entry(self, entry: Entry) -> None:
        # 保存 entry JSON 到 R2
        blob = json.dumps(entry.model_dump(), ensure_ascii=False, default=str).encode()
        _r2_put(f"{self.prefix}{entry.id}.json", blob)
        # 更新 index
        index_raw = _r2_get("index.json")
        index = json.loads(index_raw) if index_raw else []
        if entry.id not in index:
            index.append(entry.id)
        _r2_put("index.json", json.dumps(index).encode())

    def delete_entry(self, entry_id: str) -> bool:
        blob = _r2_get(f"{self.prefix}{entry_id}.json")
        if not blob:
            return False
        _r2_put(f"{self.prefix}{entry_id}.json", b"")  # R2 不支持删除空值，实际应调用 delete API
        import subprocess
        try:
            subprocess.run(["rclone", "delete", f"rag:{_R2_BUCKET}/{self.prefix}{entry_id}.json"], check=False)
        except Exception:
            pass
        index_raw = _r2_get("index.json")
        if index_raw:
            index = json.loads(index_raw)
            if entry_id in index:
                index.remove(entry_id)
                _r2_put("index.json", json.dumps(index).encode())
        return True

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """保存文本块到 KV（key: entry_id:chunk_idx, value: embedding JSON）"""
        for i, chunk in enumerate(chunks):
            ckey = f"{chunk.entry_id}:{i}"
            emb_key = f"emb:{ckey}"
            text_key = f"text:{ckey}"
            _kv_put(_KV_NS_EMBED, emb_key, json.dumps(chunk.embedding))
            _kv_put(_KV_NS_ENTRIES, text_key, chunk.text)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """从 KV 中检索最相似的文本块。"""
        texts_raw = _kv_get(_KV_NS_ENTRIES, "_all_texts") or "[]"
        try:
            all_texts = json.loads(texts_raw)
        except Exception:
            all_texts = []
        scores = []
        for idx, text in enumerate(all_texts):
            emb_raw = _kv_get(_KV_NS_EMBED, f"emb:{idx}")
            if not emb_raw:
                continue
            try:
                emb = json.loads(emb_raw)
                sim = _cosine_similarity(query_embedding, emb)
                scores.append((sim, {"entry_id": f"chunk_{idx}", "text": text}))
            except Exception:
                pass
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]


# ────────────────────────────────────────────────────────────
# Todo Store（K/V 适配）
# ────────────────────────────────────────────────────────────

_TODO_KV_NS = "todos"


def _list_todos() -> list[dict]:
    raw = _kv_get(_TODO_KV_NS, "_list") or "[]"
    try:
        return json.loads(raw)
    except Exception:
        return []


def _save_todos(todos: list[dict]) -> None:
    _kv_put(_TODO_KV_NS, "_list", json.dumps(todos, ensure_ascii=False))


def _get_todo(tid: str) -> dict | None:
    raw = _kv_get(_TODO_KV_NS, tid)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _set_todo(tid: str, todo: dict) -> None:
    _kv_put(_TODO_KV_NS, tid, json.dumps(todo, ensure_ascii=False))


def _delete_todo(tid: str) -> None:
    _kv_put(_TODO_KV_NS, tid, "")
    todos = _list_todos()
    todos = [t for t in todos if t["id"] != tid]
    _save_todos(todos)


# ────────────────────────────────────────────────────────────
# Chunk 文本切分（复用原有逻辑）
# ────────────────────────────────────────────────────────────

def _split_text(text: str, max_chars: int = 300, overlap_chars: int = 50) -> list[str]:
    import re
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    chunks = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) > max_chars and buf:
            chunks.append(buf.strip())
            buf = buf[-overlap_chars:] if overlap_chars < len(buf) else ""
        buf += " " + s if buf else s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_and_store_entry(entry: Entry) -> EntryStore:
    store = EntryStore()
    chunks_text = _split_text(entry.corrected_text)
    if not chunks_text:
        return store
    embeddings = _embed_texts(chunks_text)
    dim = len(embeddings[0]) if embeddings and embeddings[0] else 128
    chunks = [Chunk(entry_id=entry.id, text=t, start=None,
                    embedding=[0.0] * dim) for t in chunks_text]
    for c, emb in zip(chunks, embeddings):
        c.embedding = emb
    store.add_chunks(chunks)
    entry.chunks = chunks
    store.save_entry(entry)
    # 更新 KV 中的全文索引
    all_texts_raw = _kv_get(_KV_NS_ENTRIES, "_all_texts") or "[]"
    try:
        all_texts = json.loads(all_texts_raw)
    except Exception:
        all_texts = []
    all_texts.extend(chunks_text)
    _kv_put(_KV_NS_ENTRIES, "_all_texts", json.dumps(all_texts))
    return store


def search(query: str, top_k: int = 5) -> list[dict]:
    store = EntryStore()
    qv = _embed_texts([query])
    qe = qv[0] if qv else [0.0] * 128
    return store.search(qe, top_k=top_k)


# ────────────────────────────────────────────────────────────
# 知识库图谱（R2 JSON）
# ────────────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self, data_dir: str | None = None):
        self.nodes_path = "knowledge_nodes.json"
        self.edges_path = "knowledge_edges.json"
        self.nodes = self._load(self.nodes_path, KnowledgeNode)
        self.edges = self._load(self.edges_path, KnowledgeEdge)

    def _load(self, path: str, cls):
        raw = _r2_get(path)
        if not raw:
            return []
        try:
            return [cls(**item) for item in json.loads(raw)]
        except Exception:
            return []

    def save(self):
        _r2_put(self.nodes_path,
                json.dumps([n.model_dump() for n in self.nodes],
                           ensure_ascii=False, indent=2).encode())
        _r2_put(self.edges_path,
                json.dumps([e.model_dump() for e in self.edges],
                           ensure_ascii=False, indent=2).encode())

    def add_entry(self, entry_id: str, text: str):
        from app.knowledge_graph import _extract_points, _relation
        points = _extract_points(text)
        if not points:
            return
        root = self._find_or_create(points[0])
        if entry_id not in root.entries:
            root.entries.append(entry_id)
        for p in points[1:]:
            node = self._find_or_create(p)
            if entry_id not in node.entries:
                node.entries.append(entry_id)
            for src, tgt, rel, w in _relation(points[0], p):
                self._add_edge(src, tgt, rel, w)
        self.save()

    def _add_edge(self, source: str, target: str, relation: str, weight: float = 1.0):
        src = self._find_or_create(source)
        tgt = self._find_or_create(target)
        for e in self.edges:
            if e.source == src.id and e.target == tgt.id:
                e.relation = relation
                e.weight = weight
                return
        self.edges.append(KnowledgeEdge(source=src.id, target=tgt.id,
                                        relation=relation, weight=weight))

    def _find_or_create(self, name: str) -> KnowledgeNode:
        for n in self.nodes:
            if n.name == name or name in n.aliases:
                return n
        node = KnowledgeNode(id=str(uuid.uuid4()), name=name, aliases=[],
                             entries=[], summary="", parents=[], children=[])
        self.nodes.append(node)
        return node

    def list_nodes(self) -> list[KnowledgeNode]:
        return self.nodes

    def list_edges(self) -> list[KnowledgeEdge]:
        return self.edges

    def build_tree(self, root_name: str, depth: int = 3) -> dict:
        root = self._find_or_create(root_name)
        tree = {"name": root.name, "children": []}
        visited = {root.id}
        self._expand(root, tree, depth, visited)
        return tree

    def _expand(self, node: KnowledgeNode, tree_node: dict, depth: int, visited: set):
        if depth <= 0:
            return
        children = [e for e in self.edges if e.source == node.id and e.target not in visited]
        for e in children:
            child = self._get_node(e.target)
            if not child:
                continue
            visited.add(child.id)
            child_node = {"name": child.name, "relation": e.relation, "children": []}
            tree_node["children"].append(child_node)
            self._expand(child, child_node, depth - 1, visited)

    def _get_node(self, node_id: str) -> KnowledgeNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


# ────────────────────────────────────────────────────────────
# Job Queue（KV 适配）
# ────────────────────────────────────────────────────────────

class JobQueue:
    def __init__(self, data_dir: str | None = None):
        self.ns = "jobs"

    def _load(self) -> list[Job]:
        raw = _kv_get(self.ns, "_list") or "[]"
        try:
            return [Job(**item) for item in json.loads(raw)]
        except Exception:
            return []

    def save(self, jobs: list[Job]) -> None:
        _kv_put(self.ns, "_list", json.dumps([j.model_dump() for j in jobs],
                                              ensure_ascii=False, default=str, indent=2))

    def create(self, kind: str, collection_id: str = "", total: int = 0) -> Job:
        jobs = self._load()
        job = Job(id=str(uuid.uuid4()), collection_id=collection_id, kind=kind,
                  status="queued", progress=0, total=total, processed=0, failed=[])
        jobs.append(job)
        self.save(jobs)
        return job

    def get(self, job_id: str) -> Job | None:
        for j in self._load():
            if j.id == job_id:
                return j
        return None

    def list(self) -> list[Job]:
        return sorted(self._load(), key=lambda j: j.created_at, reverse=True)

    def update_progress(self, job_id: str, processed: int):
        jobs = self._load()
        for j in jobs:
            if j.id == job_id:
                j.processed = processed
                j.progress = int(processed / j.total * 100) if j.total else 100
                j.status = "done" if j.progress >= 100 else "running"
                break
        self.save(jobs)

    def mark_failed(self, job_id: str, item: str):
        jobs = self._load()
        for j in jobs:
            if j.id == job_id:
                j.failed.append(item)
                break
        self.save(jobs)


def enqueue_fav_sync(fav_id: int) -> Job:
    from app.bilibili_fav import list_fav_videos
    q = JobQueue()
    videos = list_fav_videos(fav_id)
    job = q.create("fav_sync", collection_id=str(fav_id), total=len(videos))
    return job


# ────────────────────────────────────────────────────────────
# Todo 操作函数（供 main.py 使用）
# ────────────────────────────────────────────────────────────

def list_todos() -> list[dict]:
    return _list_todos()


def create_todo(content: str, tags: list[str] = None, entry_id: str | None = None) -> dict:
    todos = _list_todos()
    todo = {
        "id": str(uuid.uuid4()),
        "content": content,
        "done": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags or [],
        "entry_id": entry_id,
    }
    todos.append(todo)
    _save_todos(todos)
    _kv_put(_TODO_KV_NS, todo["id"], json.dumps(todo, ensure_ascii=False))
    return todo


def toggle_todo(todo_id: str) -> dict | None:
    todo = _get_todo(todo_id)
    if not todo:
        return None
    todo["done"] = not todo["done"]
    _set_todo(todo_id, todo)
    _save_todos(_list_todos())
    return todo


def delete_todo(todo_id: str) -> bool:
    if not _get_todo(todo_id):
        return False
    _delete_todo(todo_id)
    return True


# ────────────────────────────────────────────────────────────
# 异步任务（音频转写：返回提示，workers 环境不执行长任务）
# ────────────────────────────────────────────────────────────

async def transcribe_media_entry(file, language: str = "auto") -> Entry:
    """Cloudflare Workers 不支持长耗时任务，返回提示。"""
    filename = file.filename or "upload"
    return Entry(
        id=str(uuid.uuid4()),
        type="netdisk_media",
        source=filename,
        title=filename,
        raw_text="[ Workers 环境暂不支持音频转写，请在本地服务器部署使用 ]",
        corrected_text="[ Workers 环境暂不支持音频转写，请在本地服务器部署使用 ]",
        status="ok",
    )
