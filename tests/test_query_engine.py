# -*- coding: utf-8 -*-
"""检索增强问答链路：上下文拼接、引用约束、出处回带。"""
from __future__ import annotations

import pytest

from app.query_engine import _build_context, answer_query


def hits() -> list[dict]:
    return [
        {'entry_id': 'e1', 'start': 0, 'text': '上班打卡早于八点三十分为正常。',
         'citation': '[e1#0]', 'rank': 0, 'score': 0.03, 'recall': 'both'},
        {'entry_id': 'e2', 'start': 12, 'text': '工号 A1024 有三次缺卡。',
         'citation': '[e2#12]', 'rank': 1, 'score': 0.02, 'recall': 'keyword'},
    ]


def patch(monkeypatch, canned='回答内容 [来源1]', captured=None):
    def fake_search(query, **kw):
        if captured is not None:
            captured['query'] = query
            captured['kw'] = kw
        return hits()

    def fake_llm(messages, temperature=None):
        if captured is not None:
            captured['messages'] = messages
            captured['temperature'] = temperature
        return canned

    monkeypatch.setattr('app.query_engine.search_hybrid', fake_search)
    monkeypatch.setattr('app.query_engine.call_chat', fake_llm)


# ------------------------------------------------------------------ 上下文拼接
def test_build_context_carries_citations():
    ctx = _build_context(hits())
    assert '[e1#0] 上班打卡早于八点三十分为正常。' in ctx
    assert '[e2#12] 工号 A1024 有三次缺卡。' in ctx
    assert ctx.count('\n\n') == 1


def test_build_context_empty():
    assert _build_context([]) == ''


# ------------------------------------------------------------------ 端到端（打桩）
def test_answer_returns_answer_and_sources(monkeypatch):
    patch(monkeypatch)
    out = answer_query('上班时间怎么算？')
    assert out['answer'] == '回答内容 [来源1]'
    assert len(out['sources']) == 2
    assert out['sources'][0]['citation'] == '[e1#0]'


def test_sources_keep_recall_provenance(monkeypatch):
    """来源里保留「哪一路召回的」，出问题能定位是向量路还是关键词路。"""
    patch(monkeypatch)
    out = answer_query('查询')
    assert {s['recall'] for s in out['sources']} == {'both', 'keyword'}


def test_prompt_forbids_speculation(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('查询')
    system = cap['messages'][0]['content']
    assert '不要推测' in system
    assert '没有相关内容' in system


def test_prompt_requests_citation(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('查询')
    assert '来源' in cap['messages'][0]['content']


def test_user_message_includes_query_and_context(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('员工工号 A1024 的情况')
    user = cap['messages'][1]['content']
    assert '员工工号 A1024 的情况' in user
    assert '[e2#12]' in user


def test_language_switch_affects_system_prompt(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('query', language='en')
    assert 'English' in cap['messages'][0]['content']


def test_top_k_and_mode_forwarded(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('查询', top_k=3, mode='weighted', alpha=0.7)
    assert cap['kw']['top_k'] == 3
    assert cap['kw']['mode'] == 'weighted'
    assert cap['kw']['alpha'] == 0.7


def test_temperature_is_moderate(monkeypatch):
    cap = {}
    patch(monkeypatch, captured=cap)
    answer_query('查询')
    assert cap['temperature'] == 0.5


def test_no_hits_still_answers_with_empty_context(monkeypatch):
    monkeypatch.setattr('app.query_engine.search_hybrid',
                        lambda q, **kw: [])
    monkeypatch.setattr('app.query_engine.call_chat',
                        lambda messages, temperature=None: '资料中没有相关内容')
    out = answer_query('不存在的问题')
    assert out['sources'] == []
    assert out['answer'] == '资料中没有相关内容'
