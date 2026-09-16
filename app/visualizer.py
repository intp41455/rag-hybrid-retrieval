import json
from app.models import Entry, Visualization
from app.llm import call_chat


def _call_llm(messages: list) -> str:
    return call_chat(messages, temperature=0.4)


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取首个 JSON 对象（容忍代码围栏/前后多余文本）。"""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"LLM 输出中无 JSON: {text[:100]}")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
        elif ch == '"' and not in_str:
            in_str = True
        elif ch == '"' and in_str:
            in_str = False
        elif ch == "{" and not in_str:
            depth += 1
        elif ch == "}" and not in_str:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 未闭合")


def visualize_entry(entry: Entry) -> Visualization:
    lang = "中文" if entry.language == "zh" else "English"
    system = f"你是可视化专家。根据{lang}文稿生成：1）markdown 思维导图源码（# 根节点 层级）；2）mermaid flowchart 源码。只输出 JSON，字段 mindmap/flowchart。"
    content = _call_llm([{"role": "system", "content": system}, {"role": "user", "content": entry.corrected_text}])
    data = _extract_json(content)
    return Visualization(mindmap=data.get("mindmap", ""), flowchart=data.get("flowchart", ""))
