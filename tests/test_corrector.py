# -*- coding: utf-8 -*-
"""字幕校对：语种判定、时间戳行解析、解析失败兜底。LLM 调用全部打桩。"""
from __future__ import annotations

import pytest

from app.corrector import PROMPTS, _detect_language, correct_entry
from app.models import Entry


def make_entry(raw: str) -> Entry:
    return Entry(id='e1', type='bilibili', source='unit-test',
                 title='字幕', raw_text=raw)


# ------------------------------------------------------------------ 语种判定
def test_detect_chinese():
    assert _detect_language('这是一段中文字幕内容') == 'zh'


def test_detect_english():
    assert _detect_language('This is an English subtitle line') == 'en'


def test_detect_empty_defaults_to_chinese():
    assert _detect_language('') == 'zh'


def test_detect_threshold_is_over_half_ascii():
    # 判定规则是 ascii 占比 > 0.5 才算英文
    assert _detect_language('ab中文') == 'zh'       # 2/4 = 0.5，不算
    assert _detect_language('abc中文') == 'en'      # 3/5 = 0.6，算


def test_prompts_exist_for_both_languages():
    assert 'zh' in PROMPTS and 'en' in PROMPTS
    assert '时间戳' in PROMPTS['zh'] or '[start,end]' in PROMPTS['zh']


# ------------------------------------------------------------------ 时间戳解析
def test_correct_entry_parses_timestamp_lines(monkeypatch):
    canned = '[0,3] 第一句话。\n[3,6] 第二句话。'
    monkeypatch.setattr('app.corrector.call_chat', lambda *a, **k: canned)
    entry = correct_entry(make_entry('原始'), language='zh')
    assert entry.corrected_text == '[0s] 第一句话。\n[3s] 第二句话。'
    assert entry.language == 'zh'


def test_correct_entry_ignores_noise_lines(monkeypatch):
    canned = ('好的，以下是校对结果：\n'
              '[10,20] 有效行。\n'
              '（以上为校对内容）\n')
    monkeypatch.setattr('app.corrector.call_chat', lambda *a, **k: canned)
    entry = correct_entry(make_entry('原始'), language='zh')
    assert entry.corrected_text == '[10s] 有效行。'


def test_correct_entry_falls_back_to_raw_when_no_timestamps(monkeypatch):
    monkeypatch.setattr('app.corrector.call_chat', lambda *a, **k: '模型没有按格式输出')
    entry = correct_entry(make_entry('原始文本'), language='zh')
    assert entry.corrected_text == '原始文本'          # 不丢内容


def test_correct_entry_auto_detect_english(monkeypatch):
    seen = {}

    def fake(messages, temperature=None):
        seen['system'] = messages[0]['content']
        return '[0,5] corrected.'

    monkeypatch.setattr('app.corrector.call_chat', fake)
    entry = correct_entry(make_entry('this is english text'), language='auto')
    assert entry.language == 'en'
    assert 'subtitle' in seen['system'].lower()


def test_correct_entry_explicit_language_skips_detection(monkeypatch):
    monkeypatch.setattr('app.corrector.call_chat', lambda *a, **k: '[0,1] ok.')
    entry = correct_entry(make_entry('This is English'), language='zh')
    assert entry.language == 'zh'


def test_correct_entry_passes_low_temperature(monkeypatch):
    seen = {}

    def fake(messages, temperature=None):
        seen['messages'] = messages
        seen['temperature'] = temperature
        return '[0,1] ok.'

    monkeypatch.setattr('app.corrector.call_chat', fake)
    correct_entry(make_entry('文本'), language='zh')
    assert seen['temperature'] == 0.3
    assert [m['role'] for m in seen['messages']] == ['system', 'user']


def test_correct_entry_handles_multiline_timestamp_text(monkeypatch):
    canned = '[0,5] 跨行的\n内容。\n[5,10] 第二段。'
    monkeypatch.setattr('app.corrector.call_chat', lambda *a, **k: canned)
    entry = correct_entry(make_entry('x'), language='zh')
    lines = entry.corrected_text.splitlines()
    assert len(lines) == 2
    assert lines[0] == '[0s] 跨行的'
