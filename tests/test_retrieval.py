# -*- coding: utf-8 -*-
"""混合召回：分词、BM25、RRF/加权融合、引用溯源。

核心要验证的一句话：**关键词路能救回向量路在专有编号上漏掉的结果。**
"""
from __future__ import annotations

import pytest

from app.retrieval import (KeywordIndex, build_keyword_index, hybrid_search,
                           make_source_ref, rrf_fuse, tokenize, weighted_fuse)


# ------------------------------------------------------------------ 分词
def test_tokenize_cjk_single_and_bigram():
    toks = tokenize('考勤规则')
    assert '考' in toks and '勤' in toks
    assert '考勤' in toks and '勤规' in toks and '规则' in toks
    assert len([t for t in toks if len(t) == 1]) == 4
    assert len([t for t in toks if len(t) == 2]) == 3


def test_tokenize_ascii_lowercased():
    assert 'a1024' in tokenize('A1024')


def test_tokenize_mixed_text():
    toks = tokenize('员工 A1024 的考勤')
    assert 'a1024' in toks
    assert '考勤' in toks


def test_tokenize_ignores_single_ascii_letter():
    toks = tokenize('a b c')
    assert toks == []


def test_tokenize_filters_cjk_stopword_unigrams():
    toks = tokenize('的考勤')
    assert '的' not in toks          # 单字虚词被过滤
    assert '考' in toks              # 实义单字保留
    assert '的考' in toks            # 双字一律保留，不做停用词判断


def test_tokenize_keeps_stopword_chars_inside_bigrams():
    # "目的" 里的 "的" 是实义，双字层不受影响
    assert '目的' in tokenize('目的')


def test_tokenize_all_stopwords_yields_empty():
    assert tokenize('的 了 是 在') == []


def test_tokenize_empty():
    assert tokenize('') == []
    assert tokenize('   ') == []


def test_tokenize_no_cjk_bigram_for_single_char():
    assert tokenize('甲') == ['甲']


# ------------------------------------------------------------------ BM25
def test_index_build_size_and_avgdl(sample_metas):
    idx = build_keyword_index(sample_metas)
    assert idx.size == 5
    assert idx._avgdl > 0
    assert len(idx._tf) == 5 and len(idx._len) == 5


def test_index_empty_is_safe():
    idx = build_keyword_index([])
    assert idx.size == 0
    assert idx.search('任意查询') == []
    assert idx.score('任意', 0) == 0.0


def test_bm25_ranks_relevant_doc_first(sample_metas):
    idx = build_keyword_index(sample_metas)
    hits = idx.search('加班', top_k=5)
    assert hits
    assert sample_metas[hits[0][0]]['start'] == 12        # 「超出部分计入加班」那条


def test_bm25_scores_positive_only(sample_metas):
    idx = build_keyword_index(sample_metas)
    assert all(s > 0 for _, s in idx.search('考勤'))


def test_bm25_no_match_returns_empty(sample_metas):
    idx = build_keyword_index(sample_metas)
    # 与语料无任何共同单字/双字
    assert idx.search('春节联欢庆典') == []
    # 纯 ASCII、无共同词
    assert idx.search('zzqwx') == []


def test_bm25_pure_stopword_query_matches_nothing(sample_metas):
    """不过滤单字虚词的话，「的」会让任意中文文本互相命中，关键词路就成了噪声源。"""
    idx = build_keyword_index(sample_metas)
    assert tokenize('的了吗呢') == []
    assert idx.search('的了吗呢') == []


def test_bm25_distinguishes_serial_numbers(sample_metas):
    """这是引入关键词路的根本原因：向量空间里 A1024 和 A2048 几乎重合。"""
    idx = build_keyword_index(sample_metas)
    a1024 = idx.search('工号 A1024', top_k=5)
    a2048 = idx.search('工号 A2048', top_k=5)
    assert a1024 and a2048
    top_1024 = sample_metas[a1024[0][0]]['text']
    top_2048 = sample_metas[a2048[0][0]]['text']
    assert 'A1024' in top_1024
    assert 'A2048' in top_2048
    assert top_1024 != top_2048


def test_bm25_lower_is_better_for_shorter_docs():
    """b 参数生效：同样命中一次，短文档得分应更高（长度归一化）。"""
    long_pad = '无关内容。' * 60
    metas = [{'entry_id': 's', 'text': '考勤规则说明。'},
             {'entry_id': 'l', 'text': '考勤规则说明。' + long_pad}]
    idx = build_keyword_index(metas)
    assert idx.score('考勤', 0) > idx.score('考勤', 1)


def test_bm25_tie_is_stable(sample_metas):
    idx = build_keyword_index(sample_metas)
    first = idx.search('考勤', top_k=5)
    assert first == idx.search('考勤', top_k=5)


# ------------------------------------------------------------------ RRF
def test_rrf_doc_in_both_lists_wins():
    fused = rrf_fuse([[1, 2, 3], [3, 4, 5]])
    order = [d for d, _ in fused]
    # 3 出现在两路里，交集文档应排最前
    assert order[0] == 3


def test_rrf_weights_bias_ranking():
    no_w = dict(rrf_fuse([[1, 2], [2, 1]]))
    assert no_w[1] == pytest.approx(no_w[2])       # 对称，同分
    biased = dict(rrf_fuse([[1, 2], [2, 1]], weights=[1.0, 0.0]))
    assert biased[1] > biased[2]


def test_rrf_empty_inputs():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_k_affects_score_magnitude_not_order():
    a = [d for d, _ in rrf_fuse([[1, 2, 3], [3, 2, 1]], k=10)]
    b = [d for d, _ in rrf_fuse([[1, 2, 3], [3, 2, 1]], k=200)]
    assert a == b


# ------------------------------------------------------------------ 加权融合
def test_weighted_fuse_normalizes_scales():
    # 第二路分数是 0~1000 量级，若不归一化会完全压制第一路
    fused = dict(weighted_fuse([[1, 2], [2, 1]], [[0.9, 0.1], [1000.0, 0.0]], alpha=0.5))
    assert fused[1] == pytest.approx(fused[2])


def test_weighted_fuse_alpha_extremes():
    hi = dict(weighted_fuse([[1, 2], [2, 1]], [[1.0, 0.0], [1.0, 0.0]], alpha=1.0))
    assert hi[1] > hi[2]
    lo = dict(weighted_fuse([[1, 2], [2, 1]], [[1.0, 0.0], [1.0, 0.0]], alpha=0.0))
    assert lo[2] > lo[1]


# ------------------------------------------------------------------ 引用
def test_make_source_ref_has_citation():
    ref = make_source_ref({'entry_id': 'e9', 'start': 30, 'text': '片段'}, 0)
    assert ref['citation'] == '[e9#30]'
    assert ref['entry_id'] == 'e9'
    assert ref['rank'] == 0


def test_make_source_ref_without_start():
    assert make_source_ref({'entry_id': 'e9', 'text': 'x'}, 1)['citation'] == '[e9]'


def test_make_source_ref_handles_missing_entry_id():
    assert make_source_ref({}, 0)['citation'] == '[unknown]'


# ------------------------------------------------------------------ 混合检索
def test_hybrid_returns_citations_and_recall_labels(sample_metas):
    idx = build_keyword_index(sample_metas)
    vec_hits = [(sample_metas[2], 0.88), (sample_metas[0], 0.42)]
    out = hybrid_search('工号 A1024', idx, vec_hits, top_k=3)
    assert out
    for h in out:
        assert h['citation'].startswith('[')
        assert h['recall'] in ('both', 'vector', 'keyword')
        assert 'score' in h


def test_hybrid_rescues_vector_miss_on_serial_number(sample_metas):
    """构造向量路「分不清 A1024 和 A2048」的场景，关键词路应把正确结果顶上来。"""
    idx = build_keyword_index(sample_metas)
    # 向量路把 A2048 排在前面（错），A1024 排第二
    vec_hits = [(sample_metas[3], 0.91), (sample_metas[2], 0.90)]
    out = hybrid_search('工号 A1024', idx, vec_hits, top_k=2)
    assert 'A1024' in out[0]['text']


def test_hybrid_pool_limits_keyword_candidates(sample_metas):
    idx = build_keyword_index(sample_metas)
    out = hybrid_search('考勤', idx, [], top_k=10, pool=1)
    assert len(out) <= 1


def test_hybrid_with_empty_vector_hits_still_works(sample_metas):
    idx = build_keyword_index(sample_metas)
    out = hybrid_search('考勤', idx, [], top_k=3)
    assert out
    assert all(h['recall'] == 'keyword' for h in out)


def test_hybrid_with_empty_index_returns_empty(sample_metas):
    idx = build_keyword_index([])
    assert hybrid_search('考勤', idx, [(sample_metas[0], 0.5)], top_k=3) == []


def test_hybrid_ignores_vector_hits_not_in_index(sample_metas):
    idx = build_keyword_index(sample_metas[:2])
    stranger = {'entry_id': 'ghost', 'start': 1, 'text': '不在索引里的片段'}
    out = hybrid_search('考勤', idx, [(stranger, 0.99)], top_k=5)
    assert all(h['entry_id'] != 'ghost' for h in out)


def test_hybrid_mode_weighted(sample_metas):
    idx = build_keyword_index(sample_metas)
    vec_hits = [(sample_metas[0], 0.9), (sample_metas[1], 0.1)]
    out = hybrid_search('上班打卡', idx, vec_hits, top_k=3, mode='weighted', alpha=0.5)
    assert out
    assert 'score' in out[0]


def test_hybrid_respects_top_k(sample_metas):
    idx = build_keyword_index(sample_metas)
    assert len(hybrid_search('考勤', idx, [], top_k=2)) == 2
    assert len(hybrid_search('考勤', idx, [], top_k=99)) <= len(sample_metas)


def test_hybrid_rank_fields_are_consistent(sample_metas):
    idx = build_keyword_index(sample_metas)
    vec_hits = [(sample_metas[0], 0.5), (sample_metas[1], 0.4)]
    out = hybrid_search('考勤', idx, vec_hits, top_k=5)
    for h in out:
        if h['recall'] == 'both':
            assert h['vector_rank'] is not None and h['keyword_rank'] is not None
        elif h['recall'] == 'vector':
            assert h['keyword_rank'] is None
        else:
            assert h['vector_rank'] is None
