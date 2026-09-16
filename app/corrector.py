import re
from app.models import Entry, Line
from app.llm import call_chat

PROMPTS = {
    "zh": "你是一名字幕校对助手。请逐句校对，保留时间戳格式 [start,end]文本。只输出校对后的文本。",
    "en": "You are a subtitle proofreading assistant. Correct fragmented subtitles into fluent sentences. Keep format [start,end]text. Output only corrected text.",
}


def _detect_language(text: str) -> str:
    ascii_chars = sum(1 for c in text if c.isascii())
    return "en" if ascii_chars / max(len(text), 1) > 0.5 else "zh"


def _call_llm(system: str, user: str) -> str:
    return call_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )


def correct_entry(entry: Entry, language: str = "auto") -> Entry:
    lang = language if language != "auto" else _detect_language(entry.raw_text)
    prompt = PROMPTS.get(lang, PROMPTS["zh"])
    corrected = _call_llm(prompt, entry.raw_text)
    lines = []
    for row in corrected.splitlines():
        m = re.match(r"\[(\d+),(\d+)\](.*)", row.strip())
        if m:
            lines.append(Line(start=int(m.group(1)), end=int(m.group(2)), text=m.group(3).strip()))
    if not lines:
        # fallback: keep raw text
        entry.corrected_text = entry.raw_text
        return entry
    entry.corrected_text = "\n".join(f"[{l.start}s] {l.text}" for l in lines)
    entry.language = lang
    return entry
