import json
import os
import numpy as np
from app.config import settings
from app.models import Entry, Chunk


def _embed(texts: list[str]) -> np.ndarray:
    # placeholder; real embedding in embed_store.py
    return np.random.rand(len(texts), 8).astype(np.float32)


class EntryStore:
    def __init__(self, data_dir: str | None = None):
        self.base = data_dir or settings.data_dir
        os.makedirs(self.base, exist_ok=True)
        self.entries_dir = os.path.join(self.base, "entries")
        self.vectors_path = os.path.join(self.base, "vectors.npy")
        self.meta_path = os.path.join(self.base, "meta.json")
        os.makedirs(self.entries_dir, exist_ok=True)
        self.vectors = self._load_vectors()
        self.meta = self._load_meta()

    def _load_vectors(self):
        if os.path.exists(self.vectors_path):
            return np.load(self.vectors_path)
        return np.zeros((0, 0), dtype=np.float32)

    def _load_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_entry(self, entry: Entry):
        path = os.path.join(self.entries_dir, f"{entry.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry.model_dump(), f, ensure_ascii=False, default=str, indent=2)

    def get_entry(self, entry_id: str) -> Entry | None:
        path = os.path.join(self.entries_dir, f"{entry_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return Entry(**json.load(f))

    def list_entries(self) -> list[Entry]:
        entries = []
        for fn in sorted(os.listdir(self.entries_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(self.entries_dir, fn), encoding="utf-8") as f:
                    entries.append(Entry(**json.load(f)))
        return entries

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)
        # 维度不一致：通常是模型切换/历史数据不兼容，清空重建
        if self.vectors.shape[0] > 0 and self.vectors.shape[1] != vectors.shape[1]:
            self.vectors = vectors
            self.meta = [{"entry_id": c.entry_id, "text": c.text, "start": c.start} for c in chunks]
            self._save()
            return
        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        for c in chunks:
            self.meta.append({"entry_id": c.entry_id, "text": c.text, "start": c.start})
        self._save()

    def search(self, query_vector: list[float], top_k: int = 5):
        qv = np.array(query_vector, dtype=np.float32)
        if self.vectors.shape[0] == 0:
            return []
        # 维度不一致（如 TF-IDF 重启后重拟合）时无法比较，返回空结果
        if qv.ndim != 1 or qv.shape[0] != self.vectors.shape[1]:
            return []
        v_norms = np.linalg.norm(self.vectors, axis=1)
        q_norm = np.linalg.norm(qv)
        denom = v_norms * q_norm
        denom[denom == 0] = 1.0
        sim = np.dot(self.vectors, qv) / denom
        idx = np.argsort(sim)[::-1][:top_k]
        return [(self.meta[i], float(sim[i])) for i in idx]

    def _save(self):
        np.save(self.vectors_path, self.vectors)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
