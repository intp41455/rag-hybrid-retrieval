import os
import pickle
import re
import numpy as np
from app.config import settings
from app.models import Entry, Chunk
from app.store import EntryStore

_model = None
_tfidf = None
_docs = []

_TFIDF_PATH = os.path.join(settings.data_dir, "tfidf.pkl")


def _get_model():
    """懒加载本地 BGE 中文向量模型（免费，离线可用）。"""
    global _model
    if _model is None:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("/tmp/bge-model")
    return _model


def _load_tfidf():
    """加载持久化的 TF-IDF（保证重启后向量维度一致）。"""
    global _tfidf, _docs
    if _tfidf is None and os.path.exists(_TFIDF_PATH):
        try:
            with open(_TFIDF_PATH, "rb") as f:
                state = pickle.load(f)
            _tfidf, _docs = state["tfidf"], state["docs"]
        except Exception:
            pass


def _save_tfidf():
    os.makedirs(settings.data_dir, exist_ok=True)
    with open(_TFIDF_PATH, "wb") as f:
        pickle.dump({"tfidf": _tfidf, "docs": _docs}, f)


def _embed(texts: list[str]) -> np.ndarray:
    # 优先使用本地 BGE 模型（语义向量）
    try:
        return _get_model().encode(texts, convert_to_numpy=True)
    except Exception:
        pass
    # Fallback: TF-IDF vectors when BGE unavailable (模型缺失/加载失败)
    from sklearn.feature_extraction.text import TfidfVectorizer
    global _tfidf, _docs
    _load_tfidf()
    if _tfidf is None or len(_tfidf.vocabulary_) == 0:
        _tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        all_texts = _docs + texts
        _tfidf.fit(all_texts)
        _save_tfidf()
    return _tfidf.transform(texts).toarray()


def _split_text(text: str, max_chars: int = 300, overlap_chars: int = 50):
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


def chunk_and_store_entry(entry: Entry, data_dir: str | None = None) -> EntryStore:
    store = EntryStore(data_dir)
    chunks_text = _split_text(entry.corrected_text)
    if not chunks_text:
        return store
    vectors = _embed(chunks_text)
    dim = vectors.shape[1] if vectors.ndim > 1 else 1
    chunks = [Chunk(entry_id=entry.id, text=t, start=None, embedding=[0.0] * dim) for t in chunks_text]
    for c, v in zip(chunks, vectors):
        c.embedding = v.tolist()
    store.add_chunks(chunks)
    entry.chunks = chunks
    _docs.extend(chunks_text)
    store.save_entry(entry)
    return store


def search(query: str, data_dir: str | None = None, top_k: int = 5):
    """纯向量召回（保留原行为，供对照/降级使用）。"""
    store = EntryStore(data_dir)
    qv = _embed([query])[0].tolist()
    return store.search(qv, top_k=top_k)


def search_hybrid(query: str, data_dir: str | None = None, top_k: int = 5,
                  pool: int = 20, mode: str = "rrf", alpha: float = 0.5):
    """向量语义召回 + BM25 关键词召回，融合排序，每条命中带 citation。

    纯向量召回在专有名词 / 编号 / 人名上会漏召回（语义空间里彼此太近），
    关键词路对精确匹配敏感，两路融合后命中明显改善。
    实现见 app/retrieval.py。
    """
    from app.retrieval import build_keyword_index, hybrid_search

    store = EntryStore(data_dir)
    qv = _embed([query])[0].tolist()
    vector_hits = store.search(qv, top_k=pool)
    kw_index = build_keyword_index(store.meta)
    return hybrid_search(query, kw_index, vector_hits,
                         top_k=top_k, pool=pool, mode=mode, alpha=alpha)

