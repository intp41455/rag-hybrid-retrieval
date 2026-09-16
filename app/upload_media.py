import os
import uuid
import tempfile
import threading
from fastapi import UploadFile
from app.models import Entry, Line

_model = None
_model_lock = threading.Lock()


def _get_model():
    """懒加载 faster-whisper 本地语音识别模型（免费，离线可用）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model


def _call_asr(audio_path: str, language: str) -> list[dict]:
    model = _get_model()
    segments, _ = model.transcribe(
        audio_path, language=language if language != "auto" else None
    )
    return [
        {"start": int(seg.start), "end": int(seg.end), "text": seg.text}
        for seg in segments
    ]


async def transcribe_media_entry(file: UploadFile, language: str = "auto") -> Entry:
    suffix = os.path.splitext(file.filename or ".mp3")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        segments = _call_asr(tmp_path, language)
        lines = [Line(start=s["start"], end=s["end"], text=s["text"]) for s in segments]
        raw_text = "\n".join(f"[{l.start}s] {l.text}" for l in lines)
        return Entry(
            id=str(uuid.uuid4()), type="netdisk_media", source=file.filename or "",
            title=file.filename or "", raw_text=raw_text, corrected_text=raw_text, status="ok"
        )
    finally:
        os.unlink(tmp_path)
