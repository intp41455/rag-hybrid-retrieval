import io
import re
import struct
import wave


def split_text_for_tts(text: str) -> list[str]:
    raw = re.split(r"(?<=[。！？.!?])\s*", text)
    return [r.strip() for r in raw if r.strip()]


def _dummy_wav(text: str) -> bytes:
    """Generate a minimal valid WAV file as fallback when no TTS engine is available."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        # Generate ~0.5s of silence as placeholder
        frames = b"\x00\x00" * 22050
        wf.writeframes(frames)
    return buf.getvalue()


def synthesize(text: str, language: str = "zh") -> bytes:
    """本地 TTS 语音合成（免费，离线可用）。返回 WAV 字节。"""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        if language == "zh":
            engine.setProperty("voice", "zh")
        else:
            engine.setProperty("voice", "en")
        with io.BytesIO() as buf:
            engine.save_to_file(text, buf)
            engine.runAndWait()
            buf.seek(0)
            return buf.read()
    except Exception:
        return _dummy_wav(text)
