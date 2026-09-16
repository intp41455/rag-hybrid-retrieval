"""统一的大模型调用入口（免费方案：Agnes AI，OpenAI 兼容接口）。

所有需要 LLM 的模块（分析、校对、可视化、知识图谱、问答）都走这里，
只改这一处即可切换模型/平台。
"""
import httpx
from app.config import settings


def call_chat(messages: list, temperature: float = 0.5) -> str:
    """调用 LLM 对话补全（OpenAI 兼容接口，默认 Agnes AI），返回模型输出的文本。"""
    if not settings.agnes_api_key:
        raise ValueError("AGNES_API_KEY 未设置，请在 .env 中配置")
    resp = httpx.post(
        settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.agnes_api_key}"},
        json={"model": settings.llm_model, "messages": messages, "temperature": temperature},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
