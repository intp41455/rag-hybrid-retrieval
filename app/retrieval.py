# -*- coding: utf-8 -*-
"""混合召回：向量语义召回 + 关键词召回，融合排序。

为什么需要混合——
  纯向量召回在**专有名词 / 编号 / 人名 / 型号**上会漏召回：
  这些词的语义空间距离很近，问 "工号 A1024 的记录" 和 "工号 A2048 的记录"
  在向量空间里几乎重合，模型区分不开。
  而关键词召回（BM25）对这类精确匹配非常敏感，但不懂同义改写。
  两路并行召回再融合，是生产上的常见做法。

融合策略提供两种，默认 RRF：
  - RRF（Reciprocal Rank Fusion）：只看名次不看分数，天然免疫两路分数不可比的问题
    （余弦相似度 ∈ [-1,1]，BM25 是无上界实数，直接加权需要归一化，容易失真）
  - weighted：min-max 归一化后按 alpha 加权求和，可调两路权重

中文分词不引 jieba：用「CJK 单字 + 相邻双字」的字符 n-gram。
零依赖、对未登录词（人名/编号/专名）友好，代价是索引略大。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal

CJK_RUN = re.compile(r'[\u4e00-\u9fff]+')
ASCII_WORD = re.compile(r'[A-Za-z0-9_]+')

DEFAULT_RRF_K = 60

# 单字层的高频虚词。不过滤的话，「的」「了」会让任意两条中文文本互相"命中"，
# 关键词路就退化成噪声源了。只过滤单字，双字词（如"目的"）一律保留。
CJK_UNIGRAM_STOP = set(
    '的了是在和与就都而及或等这那有无不也被把对为以之其中个我你他她它们着过么呢吧啊哦嗯吗'
)


# ------------------------------------------------------------------ 分词
def tokenize(text: str) -> list[str]:
    """CJK 走字符 1-gram + 2-gram；ASCII 走小写词。零依赖。"""
    if not text:
        return []
    tokens: list[str] = []

    for run in CJK_RUN.findall(text):
        tokens.extend(c for c in run if c not in CJK_UNIGRAM_STOP)      # 单字（去虚词）
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            # 双字里"至少一个字是实义"才保留："目的" 留、"的了" 丢
            if bg[0] not in CJK_UNIGRAM_STOP or bg[1] not in CJK_UNIGRAM_STOP:
                tokens.append(bg)

    for w in ASCII_WORD.findall(text):
        w = w.lower()
        if len(w) > 1 or w.isdigit():
            tokens.append(w)

    return tokens


# ------------------------------------------------------------------ BM25
@dataclass
class KeywordIndex:
    """标准 BM25（k1 控制词频饱和，b 控制文档长度归一化强度）。"""
    k1: float = 1.5
    b: float = 0.75
    docs: list[dict] = field(default_factory=list)
    _tf: list[Counter] = field(default_factory=list)
    _len: list[int] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0

    # ---------- 构建 ----------
    @classmethod
    def build(cls, metas: Iterable[dict], **kw) -> 'KeywordIndex':
        idx = cls(**kw)
        idx.docs = [dict(m) for m in metas]
        idx._tf, idx._len = [], []
        idx._df = Counter()
        for d in idx.docs:
            tf = Counter(tokenize(d.get('text') or ''))
            idx._tf.append(tf)
            idx._len.append(sum(tf.values()))
            idx._df.update(tf.keys())
        idx._avgdl = (sum(idx._len) / len(idx._len)) if idx._len else 0.0
        return idx

    @property
    def size(self) -> int:
        return len(self.docs)

    # ---------- 检索 ----------
    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, i: int) -> float:
        if not self._tf or i >= len(self._tf):
            return 0.0
        tf, dl = self._tf[i], self._len[i]
        avgdl = self._avgdl or 1.0
        total = 0.0
        for term in set(tokenize(query)):
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
            total += self._idf(term) * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.docs:
            return []
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))       # 同分按索引稳定排序
        return scored[:top_k]


# ------------------------------------------------------------------ 融合
def rrf_fuse(rankings: list[list[int]], k: int = DEFAULT_RRF_K,
             weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score = Σ w_i / (k + rank_i)。

    只依赖名次，所以不需要两路分数可比 —— 这是它比加权求和更稳的原因。
    """
    weights = weights or [1.0] * len(rankings)
    acc: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking):
            acc[doc_id] = acc.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_fuse(rankings: list[list[int]],
                  score_lists: list[list[float]],
                  alpha: float = 0.5) -> list[tuple[int, float]]:
    """两路分数各自 min-max 归一化后加权求和。alpha 是向量的权重。

    归一化是必需的：余弦相似度在 [-1,1]，BM25 是无上界实数，直接相加会被 BM25 吃掉。
    """
    totals: dict[int, float] = {}
    for i, (ranking, scores) in enumerate(zip(rankings, score_lists)):
        w = alpha if i == 0 else (1 - alpha)
        norm = _minmax(scores)
        for doc_id, ns in zip(ranking, norm):
            totals[doc_id] = totals.get(doc_id, 0.0) + w * ns
    return sorted(totals.items(), key=lambda x: (-x[1], x[0]))


# ------------------------------------------------------------------ 引用
def make_source_ref(meta: dict, rank: int) -> dict:
    """每条命中必须带可回查的出处，从机制上抑制模型编造。"""
    entry_id = meta.get('entry_id') or 'unknown'
    start = meta.get('start')
    locator = f"{entry_id}#{start}" if start is not None else str(entry_id)
    return {
        'entry_id': entry_id,
        'start': start,
        'text': meta.get('text') or '',
        'rank': rank,
        'citation': f'[{locator}]',
    }


# ------------------------------------------------------------------ 对外入口
def hybrid_search(query: str, keyword_index: KeywordIndex,
                  vector_hits: list[tuple[dict, float]],
                  top_k: int = 5, pool: int = 20,
                  mode: Literal['rrf', 'weighted'] = 'rrf',
                  alpha: float = 0.5,
                  rrf_k: int = DEFAULT_RRF_K) -> list[dict]:
    """两路召回 -> 融合 -> 取 top_k。

    vector_hits 由调用方（embed_store.search）传入，便于离线测试与替换检索后端。
    返回的每一项都带 citation，可直接回查原文。
    """
    # 向量路的 meta 与关键词索引的 docs 是同一份数据，用 (entry_id, start, text) 定位
    def key_of(m: dict) -> tuple:
        return (m.get('entry_id'), m.get('start'), (m.get('text') or '')[:64])

    kw_pairs = keyword_index.search(query, top_k=pool)
    kw_ids = [i for i, _ in kw_pairs[:pool]]

    vec_ids: list[int] = []
    vec_scores: list[float] = []
    pos = {key_of(d): i for i, d in enumerate(keyword_index.docs)}
    kw_scores: list[float] = []
    for m, s in vector_hits:
        i = pos.get(key_of(m))
        if i is None:
            continue
        vec_ids.append(i)
        vec_scores.append(float(s))
    kw_scores = [s for _, s in kw_pairs]

    if mode == 'weighted':
        fused = weighted_fuse([vec_ids, kw_ids], [vec_scores, kw_scores], alpha=alpha)
    else:
        fused = rrf_fuse([vec_ids, kw_ids], k=rrf_k)

    kw_rank_map = {i: r for r, (i, _) in enumerate(kw_pairs)}
    vec_rank_map = {i: r for r, i in enumerate(vec_ids)}

    out: list[dict] = []
    for rank, (doc_id, score) in enumerate(fused[:top_k]):
        if doc_id >= len(keyword_index.docs):
            continue
        ref = make_source_ref(keyword_index.docs[doc_id], rank)
        ref.update({
            'score': round(float(score), 6),
            'vector_rank': vec_rank_map.get(doc_id),
            'keyword_rank': kw_rank_map.get(doc_id),
            'recall': ('both' if doc_id in vec_rank_map and doc_id in kw_rank_map
                       else 'vector' if doc_id in vec_rank_map else 'keyword'),
        })
        out.append(ref)
    return out


def build_keyword_index(metas: Iterable[dict]) -> KeywordIndex:
    return KeywordIndex.build(metas)
