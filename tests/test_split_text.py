# -*- coding: utf-8 -*-
"""切片策略：300 字 / 50 字重叠。切片质量直接决定召回上限。"""
from __future__ import annotations

from app.embed_store import _split_text


def test_short_text_single_chunk():
    assert _split_text('只有一句话。') == ['只有一句话。']


def test_empty_text_returns_empty():
    assert _split_text('') == []
    assert _split_text('   ') == []


def test_splits_at_sentence_boundary():
    text = '第一句话。' * 40          # 200 字
    chunks = _split_text(text, max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    for c in chunks:
        # 不应把句子从中间劈开
        assert c.endswith('。') or c.endswith('！') or c.endswith('？')


def test_respects_max_chars_roughly():
    text = '这是一句长度适中的句子。' * 50
    chunks = _split_text(text, max_chars=120, overlap_chars=30)
    # 允许最后一句略微超出，但不该翻倍
    assert all(len(c) <= 120 + 20 for c in chunks)


def test_overlap_creates_shared_context():
    text = '句子甲内容。句子乙内容。句子丙内容。句子丁内容。句子戊内容。'
    chunks = _split_text(text, max_chars=15, overlap_chars=8)
    assert len(chunks) >= 2
    # 相邻块之间有重叠，避免关键句被切断后两边都拿不到完整语义
    assert chunks[0][-8:] in chunks[1] or chunks[1][:8] in chunks[0]


def test_overlap_larger_than_buffer_is_safe():
    """overlap >= 缓冲区长度时不应崩，也不应无限循环。"""
    chunks = _split_text('甲乙丙。丁戊己。庚辛壬。', max_chars=3, overlap_chars=999)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_english_punctuation_supported():
    text = 'First sentence. Second sentence. Third sentence.'
    chunks = _split_text(text, max_chars=20, overlap_chars=5)
    assert len(chunks) >= 2


def test_no_empty_chunks():
    chunks = _split_text('甲。  。乙。  丙。', max_chars=50)
    assert all(c.strip() for c in chunks)


def test_all_content_preserved():
    """所有句子都应出现在至少一个切片中（不丢内容）。"""
    sentences = [f'第{i}句内容。' for i in range(30)]
    text = ''.join(sentences)
    joined = ' '.join(_split_text(text, max_chars=60, overlap_chars=10))
    for s in sentences:
        assert s in joined
