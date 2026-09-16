from app.embed_store import search_hybrid
from app.llm import call_chat


def _call_llm(messages: list) -> str:
    return call_chat(messages, temperature=0.5)


def _build_context(hits: list[dict]) -> str:
    """每条片段都带上 citation 标记，要求模型引用。"""
    return "\n\n".join(f"{h['citation']} {h['text']}" for h in hits)


def answer_query(query: str, language: str = "zh", top_k: int = 5,
                 mode: str = "rrf", alpha: float = 0.5) -> dict:
    """检索增强问答：混合召回 -> 拼上下文 -> 生成 -> 回带出处。

    sources 里保留每条命中的 citation / recall 路径，便于事后核对模型有没有跑偏。
    """
    hits = search_hybrid(query, top_k=top_k, mode=mode, alpha=alpha)
    context = _build_context(hits)
    lang = "中文" if language == "zh" else "English"
    system = (
        f"你是知识库问答助手。请根据{lang}片段回答问题。"
        "只依据给出的片段作答；片段不足时明确说「资料中没有相关内容」，不要推测。"
        "回答中涉及事实的句子请标注片段编号，格式如 [来源1]。"
    )
    content = _call_llm([
        {"role": "system", "content": system},
        {"role": "user", "content": f"问题：{query}\n\n片段：\n{context}"},
    ])
    return {"answer": content, "sources": hits}
