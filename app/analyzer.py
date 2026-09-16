import json
from app.models import Entry, Analysis
from app.llm import call_chat
from app.visualizer import _extract_json


def _call_llm(messages: list) -> str:
    return call_chat(messages, temperature=0.5)


def _flatten(items) -> list[str]:
    """LLM 有时返回对象数组（如 {"topic":..., "detail":...}），统一展平为字符串列表。"""
    out = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            parts = [str(v) for v in it.values() if v]
            out.append("：".join(parts) if parts else str(it))
        else:
            out.append(str(it))
    return out


def analyze_entry(entry: Entry) -> Analysis:
    lang = "中文" if entry.language == "zh" else "English"
    system = f"你是一名学习助手。请根据以下{lang}文稿生成：1）内容总结；2）知识点拆解（列表）；3）扩展联想（列表）。只输出 JSON，字段 summary/knowledge/expansion。"
    content = _call_llm([{"role": "system", "content": system}, {"role": "user", "content": entry.corrected_text}])
    data = _extract_json(content)
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    return Analysis(summary=summary, knowledge=_flatten(data.get("knowledge")), expansion=_flatten(data.get("expansion")))
