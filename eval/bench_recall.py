# -*- coding: utf-8 -*-
"""混合召回对比实验：纯向量 / 纯 BM25 / RRF 融合 / 加权融合。

产出 MRR@5 / Hit@5 / Recall@10，用于给「加了 BM25 之后正确结果名次前移」
这句定性描述补上量化数据。

用法：
    python eval/bench_recall.py

语料：项目 data/ 下的真实条目（去重后按 300 字 / 50 字重叠切分为 chunk）。
查询：从每个 chunk 中抽取一个子句作为 query，正例是该 chunk 本身。
     这是无标注语料下常用的自监督构造法，不引入外部数据。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# 让脚本能直接 import 被测代码（真实实现，不是重写一份）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import (  # noqa: E402
    KeywordIndex,
    build_keyword_index,
    hybrid_search,
    rrf_fuse,
    weighted_fuse,
)

DATA_DIR = os.environ.get("RAG_DATA_DIR", r"D:\rag-knowledge-project\data")
MODEL_DIR = os.environ.get("BGE_MODEL_DIR", r"D:\models\bge-small-zh-v1.5")

CHUNK = 300
OVERLAP = 50
TOP_K = 5


# ------------------------------------------------------------------ 语料
def load_corpus():
    """读条目 -> 按 title 去重 -> 切分为 chunk。"""
    entries = []
    for f in sorted(Path(DATA_DIR).glob("entries/*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = (j.get("corrected_text") or j.get("raw_text") or "").strip()
        if len(text) < 10:
            continue
        entries.append({"id": j["id"], "title": j.get("title") or "", "text": text})

    # 去重：同一 title + 同一长度视为重复导入
    seen, uniq = set(), []
    for e in entries:
        key = (e["title"], len(e["text"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)

    chunks = []
    for e in uniq:
        t = e["text"]
        step = CHUNK - OVERLAP
        for i in range(0, len(t), step):
            seg = t[i:i + CHUNK].strip()
            if len(seg) < 30:
                continue
            chunks.append({
                "id": f'{e["id"]}#{len(chunks)}',
                "text": seg,
                "title": e["title"],
                "entry_id": e["id"],
            })
        if len(t) < CHUNK:
            continue
    return uniq, chunks


# ------------------------------------------------------------- 查询构造
CJK = re.compile(r"[\u4e00-\u9fff]")


def make_query(chunk_text: str, maxlen: int = 24) -> str:
    """从 chunk 中截一个完整子句作为 query。

    优先取首个句号/问号前的完整句；太长则按 maxlen 截断到标点处。
    """
    t = chunk_text.strip()
    m = re.search(r"^[^。！？\n]{8,80}[。！？]", t)
    if m:
        return m.group(0)
    seg = t[:maxlen * 2]
    m = re.search(r"^.{6,}?[，,、；;：:。！？\n]", seg)
    return (m.group(0) if m else seg)[:maxlen * 2].strip()


# ------------------------------------------------------------------ 指标
def mrr_at_k(hits, gold, k=TOP_K):
    for i, h in enumerate(hits[:k], 1):
        if h == gold:
            return 1.0 / i, i
    return 0.0, None


def evaluate(rank_fn, query_idx, golds, k=TOP_K):
    mrrs, hits, recalls, ranks = [], [], [], []
    for q, gold in zip(query_idx, golds):
        ranked = rank_fn(q)
        m, rank = mrr_at_k(ranked, gold, k)
        mrrs.append(m)
        hits.append(1.0 if rank is not None else 0.0)
        # Recall@10：正例是否进入前 10
        recalls.append(1.0 if gold in ranked[:10] else 0.0)
        # 未命中记为 999，会拉高均值；另记一份「命中时」的平均名次
        abs_rank = 1 + ranked.index(gold) if gold in ranked else 999
        ranks.append(abs_rank)
    n = len(golds)
    hit_ranks = [r for r in ranks if r != 999]
    return {
        "MRR@5": sum(mrrs) / n,
        "Hit@5": sum(hits) / n,
        "Recall@10": sum(recalls) / n,
        "avg_rank_when_hit": (sum(hit_ranks) / len(hit_ranks)) if hit_ranks else 999.0,
        "miss_rate": (n - len(hit_ranks)) / n,
    }


# ------------------------------------------------------------------ 主流程
def main():
    uniq, chunks = load_corpus()
    print(f"去重后文档 {len(uniq)} 篇 -> 切分为 {len(chunks)} 个 chunk"
          f"（{CHUNK} 字 / {OVERLAP} 字重叠）")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("缺少 sentence-transformers:", e)
        return 1

    print(f"加载本地模型 {MODEL_DIR} ...")
    model = SentenceTransformer(MODEL_DIR)
    print(f"向量维度 = {model.get_sentence_embedding_dimension()}")

    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, batch_size=16, show_progress_bar=False,
                        normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    metas = [{"id": c["id"], "text": c["text"], "entry_id": c["entry_id"]}
             for c in chunks]
    kw = build_keyword_index(metas)
    ids = [c["id"] for c in chunks]

    # 每个 chunk 一条 query，正例是该 chunk（用下标做统一标识，与检索代码一致）
    pairs = []          # (query, chunk_index)
    for i, c in enumerate(chunks):
        q = make_query(c["text"])
        if len(CJK.findall(q)) < 4:      # 太短/无实义的跳过
            continue
        pairs.append((q, i))
    queries = [q for q, _ in pairs]
    golds = [i for _, i in pairs]
    print(f"构造查询 {len(queries)} 条（每个 chunk 一条，正例即该 chunk）\n")

    qv = model.encode(queries, batch_size=16, show_progress_bar=False,
                      normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")

    n_docs = len(chunks)

    def rank_vector(i):
        sims = vecs @ qv[i]
        return [int(j) for j in np.argsort(-sims)]

    def rank_bm25(i):
        return [int(j) for j, _ in kw.search(queries[i], top_k=n_docs)]

    def rank_rrf(i):
        return [int(j) for j, _ in rrf_fuse([rank_vector(i), rank_bm25(i)], k=60)]

    def rank_weighted(i):
        sims = vecs @ qv[i]
        v_order = [int(j) for j in np.argsort(-sims)]
        v_scores = [float(sims[j]) for j in v_order]
        b_pairs = kw.search(queries[i], top_k=n_docs)
        b_order = [int(j) for j, _ in b_pairs]
        b_scores = [float(s) for _, s in b_pairs]
        return [int(j) for j, _ in
                weighted_fuse([v_order, b_order], [v_scores, b_scores], alpha=0.5)]

    rows = []
    for name, fn in [
        ("纯向量语义", rank_vector),
        ("纯 BM25 关键词", rank_bm25),
        ("混合 · RRF(k=60)", rank_rrf),
        ("混合 · 加权(α=0.5)", rank_weighted),
    ]:
        r = evaluate(fn, range(len(queries)), golds)
        rows.append((name, r))
        print(f"{name:<20} MRR@5={r['MRR@5']:.4f}  Hit@5={r['Hit@5']:.4f}  "
              f"Recall@10={r['Recall@10']:.4f}  命中时平均名次={r['avg_rank_when_hit']:.2f}  "
              f"未进全库={r['miss_rate']*100:.1f}%")

    # 保存结果
    out = Path(__file__).parent / "recall_bench_result.json"
    out.write_text(json.dumps({
        "docs": len(uniq),
        "chunks": len(chunks),
        "queries": len(queries),
        "chunk_size": CHUNK,
        "overlap": OVERLAP,
        "rows": [{"method": n, **r} for n, r in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")

    base = rows[0][1]["MRR@5"]
    best = max(rows[2:], key=lambda x: x[1]["MRR@5"])
    if base > 0:
        print(f"\n相对纯向量，{best[0]} 的 MRR@5 提升 "
              f"{(best[1]['MRR@5'] - base) / base * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
