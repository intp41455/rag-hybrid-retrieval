import json
import os
import re
import uuid
from app.config import settings
from app.models import KnowledgeNode, KnowledgeEdge
from app.llm import call_chat


class KnowledgeGraph:
    def __init__(self, data_dir: str | None = None):
        self.dir = data_dir or settings.data_dir
        self.nodes_path = os.path.join(self.dir, "knowledge_nodes.json")
        self.edges_path = os.path.join(self.dir, "knowledge_edges.json")
        os.makedirs(self.dir, exist_ok=True)
        self.nodes = self._load(self.nodes_path, KnowledgeNode)
        self.edges = self._load(self.edges_path, KnowledgeEdge)

    def _load(self, path: str, cls):
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [cls(**item) for item in json.load(f)]

    def save(self):
        with open(self.nodes_path, "w", encoding="utf-8") as f:
            json.dump([n.model_dump() for n in self.nodes], f, ensure_ascii=False, indent=2)
        with open(self.edges_path, "w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in self.edges], f, ensure_ascii=False, indent=2)

    def add_entry(self, entry_id: str, text: str):
        """抽取知识点，关联到条目，并构建知识树（首个为根/主题，其余为子知识点）"""
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
        self.edges.append(KnowledgeEdge(source=src.id, target=tgt.id, relation=relation, weight=weight))

    def _find_or_create(self, name: str) -> KnowledgeNode:
        for n in self.nodes:
            if n.name == name or name in n.aliases:
                return n
        node = KnowledgeNode(id=str(uuid.uuid4()), name=name, aliases=[], entries=[], summary="", parents=[], children=[])
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


def _extract_points(text: str) -> list[str]:
    """由 LLM 抽取核心知识点；失败时降级为关键词切分"""
    try:
        content = call_chat(
            [
                {"role": "system", "content": "提取文本中的核心知识点，返回 JSON 数组，如 [\"机器学习\",\"神经网络\"]，只输出数组。"},
                {"role": "user", "content": text[:2000]},
            ],
            temperature=0.2,
        )
        content = content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception:
        return _keyword_split(text)


def _keyword_split(text: str) -> list[str]:
    """兜底：取出现频率较高的中文词（2-4字），最多10个"""
    words = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:10]]


def _relation(a: str, b: str) -> list[tuple[str, str, str, float]]:
    """判定两个知识点之间的关联关系；失败时兜底为“相关”"""
    try:
        content = call_chat(
            [
                {"role": "system", "content": "判断知识点A和B的关系，返回 JSON，如 [\"A\",\"B\",\"前置\",0.9]，关系取：前置/包含/相关/扩展/对比。只输出数组。"},
                {"role": "user", "content": f"A={a}\nB={b}"},
            ],
            temperature=0.2,
        )
        data = json.loads(content.strip().replace("```json", "").replace("```", "").strip())
        return [(str(data[0]), str(data[1]), str(data[2]), float(data[3]))]
    except Exception:
        return [(a, b, "相关", 0.5)]
