import io
import uuid
import chardet
import pysrt
from fastapi import UploadFile
from app.models import Entry, Line


class DocParseError(Exception):
    pass


async def parse_uploaded_doc_entry(file: UploadFile) -> Entry:
    raw = await file.read()
    if not raw:
        raise DocParseError("空文件")
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    text = raw.decode(encoding, errors="replace")
    garbled_ratio = text.count("�") / max(len(text), 1)
    if detected.get("confidence", 0) < 0.5 or garbled_ratio > 0.05:
        return Entry(
            id=str(uuid.uuid4()), type="netdisk_doc", source=file.filename or "",
            title=file.filename or "", raw_text="", corrected_text="",
            status="encoding_error", error_message="文稿编码异常"
        )
    ext = (file.filename or "").lower().split(".")[-1]
    if ext == "srt":
        subs = pysrt.from_string(text)
        lines = [Line(start=int(sub.start.ordinal/1000), end=int(sub.end.ordinal/1000), text=sub.text.replace("\n", " ")) for sub in subs]
    elif ext == "vtt":
        lines = _parse_vtt(text)
    else:
        lines = _parse_plain_text(text)
    raw_text = "\n".join(f"[{l.start}s] {l.text}" for l in lines)
    return Entry(
        id=str(uuid.uuid4()), type="netdisk_doc", source=file.filename or "",
        title=file.filename or "", raw_text=raw_text, corrected_text=raw_text, status="ok"
    )


def _parse_vtt(text: str) -> list[Line]:
    lines = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        parts = block.split("\n")
        if len(parts) >= 2 and " --> " in parts[0]:
            start_str = parts[0].split(" --> ")[0]
            start = _vtt_time_to_seconds(start_str)
            content = " ".join(parts[1:])
            lines.append(Line(start=start, end=start+5, text=content))
    return lines


def _vtt_time_to_seconds(s: str) -> int:
    h, m, sec = s.replace(",", ".").split(":")
    return int(float(h)*3600 + float(m)*60 + float(sec))


def _parse_plain_text(text: str) -> list[Line]:
    out = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if line:
            out.append(Line(start=i, end=i+1, text=line))
    return out
