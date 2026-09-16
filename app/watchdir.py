import os

MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".mp3", ".wav", ".m4a", ".aac"}


def scan_watchdir(path: str) -> list[str]:
    """返回目录下所有音视频文件绝对路径（递归）"""
    out = []
    if not os.path.isdir(path):
        return out
    for root, _dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in MEDIA_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)
