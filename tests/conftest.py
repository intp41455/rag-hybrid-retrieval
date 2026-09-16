# -*- coding: utf-8 -*-
"""测试夹具：把数据目录隔离到 tmp，避免污染 ./data。"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def data_dir(tmp_path) -> str:
    d = tmp_path / 'data'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def sample_metas() -> list[dict]:
    """一段中文语料 + 两条含专有编号的记录（用来验证关键词路的价值）。"""
    return [
        {'entry_id': 'e1', 'start': 0,
         'text': '考勤核算规则：上班打卡时间早于八点三十分视为正常出勤。'},
        {'entry_id': 'e1', 'start': 12,
         'text': '下班打卡时间晚于十七点三十分视为正常下班，超出部分计入加班。'},
        {'entry_id': 'e2', 'start': 0,
         'text': '员工工号 A1024 的月度考勤存在三次缺卡记录，需要人工复核。'},
        {'entry_id': 'e2', 'start': 20,
         'text': '员工工号 A2048 的月度考勤全部正常，无异常记录。'},
        {'entry_id': 'e3', 'start': 0,
         'text': '公文格式标准化涉及字体、字号、行距与缩进的多层继承关系。'},
    ]
