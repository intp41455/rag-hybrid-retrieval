# -*- coding: utf-8 -*-
"""存储层：持久化、维度不兼容处理、相似度排序。"""
from __future__ import annotations

import numpy as np
import pytest

from app.models import Chunk, Entry
from app.store import EntryStore


def make_chunk(entry_id: str, text: str, vec: list[float], start: int | None = 0) -> Chunk:
    return Chunk(entry_id=entry_id, text=text, start=start, embedding=vec)


def sample_entry(eid: str = 'e1', title: str = '测试条目') -> Entry:
    return Entry(id=eid, type='note', source='unit-test', title=title, raw_text='原始')


# ------------------------------------------------------------------ 空态
def test_new_store_is_empty(data_dir):
    s = EntryStore(data_dir)
    assert s.vectors.shape[0] == 0
    assert s.meta == []
    assert s.search([0.1, 0.2]) == []
    assert s.list_entries() == []


def test_add_empty_chunks_is_noop(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([])
    assert s.vectors.shape[0] == 0


# ------------------------------------------------------------------ 写入与检索
def test_add_and_search_orders_by_similarity(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([
        make_chunk('e1', '甲', [1.0, 0.0]),
        make_chunk('e1', '乙', [0.0, 1.0]),
    ])
    hits = s.search([1.0, 0.0], top_k=2)
    assert [h[0]['text'] for h in hits] == ['甲', '乙']
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_top_k_limits(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', f't{i}', [1.0, float(i)]) for i in range(5)])
    assert len(s.search([1.0, 1.0], top_k=2)) == 2
    assert len(s.search([1.0, 1.0], top_k=99)) == 5


def test_search_meta_matches_vector_order(data_dir):
    s = EntryStore(data_dir)
    chunks = [make_chunk('e1', '甲', [1.0, 0.0]), make_chunk('e2', '乙', [0.0, 1.0])]
    s.add_chunks(chunks)
    hits = s.search([0.0, 1.0], top_k=2)
    assert hits[0][0]['entry_id'] == 'e2'


def test_zero_vector_does_not_divide_by_zero(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', '零向量', [0.0, 0.0])])
    hits = s.search([0.0, 0.0], top_k=1)
    assert hits and np.isfinite(hits[0][1])


# ------------------------------------------------------------------ 维度不兼容
def test_dimension_mismatch_returns_empty_rather_than_garbage(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', '甲', [1.0, 0.0, 0.0])])
    assert s.search([1.0, 0.0]) == []          # 2 维 query 查 3 维索引


def test_add_chunks_with_new_dim_rebuilds_index(data_dir):
    """切换 embedding 模型后维度会变，此时应重建索引而不是拼坏矩阵。"""
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', '旧模型', [1.0, 0.0])])
    assert s.vectors.shape == (1, 2)

    s.add_chunks([make_chunk('e2', '新模型', [1.0, 0.0, 0.0])])
    assert s.vectors.shape == (1, 3)
    assert [m['entry_id'] for m in s.meta] == ['e2']


def test_same_dim_appends(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', '甲', [1.0, 0.0])])
    s.add_chunks([make_chunk('e2', '乙', [0.0, 1.0])])
    assert s.vectors.shape == (2, 2)
    assert len(s.meta) == 2


# ------------------------------------------------------------------ 持久化
def test_vectors_and_meta_survive_reload(data_dir):
    s = EntryStore(data_dir)
    s.add_chunks([make_chunk('e1', '甲', [1.0, 0.0]), make_chunk('e1', '乙', [0.0, 1.0])])

    s2 = EntryStore(data_dir)
    assert s2.vectors.shape == (2, 2)
    assert len(s2.meta) == 2
    assert s2.search([1.0, 0.0], top_k=1)[0][0]['text'] == '甲'


def test_entry_roundtrip(data_dir):
    s = EntryStore(data_dir)
    e = sample_entry('e42', '考勤条目')
    s.save_entry(e)
    got = s.get_entry('e42')
    assert got is not None
    assert got.id == 'e42' and got.title == '考勤条目' and got.raw_text == '原始'


def test_get_missing_entry_returns_none(data_dir):
    assert EntryStore(data_dir).get_entry('nope') is None


def test_list_entries_sorted_and_complete(data_dir):
    s = EntryStore(data_dir)
    for eid in ('c', 'a', 'b'):
        s.save_entry(sample_entry(eid))
    assert [e.id for e in s.list_entries()] == ['a', 'b', 'c']


def test_entry_json_is_utf8_not_escaped(data_dir):
    s = EntryStore(data_dir)
    s.save_entry(sample_entry('e1', '中文标题'))
    import os
    raw = open(os.path.join(s.entries_dir, 'e1.json'), encoding='utf-8').read()
    assert '中文标题' in raw            # 不是 \u4e2d\u6587
