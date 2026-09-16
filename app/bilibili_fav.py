import httpx
from app.config import settings


def _headers() -> dict:
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com"}
    if settings.bilibili_sessdata:
        headers["Cookie"] = f"SESSDATA={settings.bilibili_sessdata}"
    return headers


def _get_mid() -> int:
    """通过 SESSDATA 获取当前登录用户的 mid"""
    url = "https://api.bilibili.com/x/web-interface/nav"
    r = httpx.get(url, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0 or not data.get("data", {}).get("mid"):
        raise ValueError(f"获取B站用户信息失败（SESSDATA可能无效）: {data.get('message')}")
    return data["data"]["mid"]


def list_favorites() -> list[dict]:
    """获取当前用户创建的收藏夹列表"""
    url = "https://api.bilibili.com/x/v3/fav/folder/created/list-all"
    r = httpx.get(url, params={"up_mid": _get_mid()}, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    return data.get("list", []) or []


def list_fav_videos(fav_id: int, page: int = 1, page_size: int = 20) -> list[dict]:
    """获取收藏夹内的视频列表"""
    url = "https://api.bilibili.com/x/v3/fav/resource/list"
    params = {"media_id": fav_id, "pn": page, "ps": page_size}
    r = httpx.get(url, params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()["data"].get("medias", []) or []


def fetch_multi_page(bvid: str) -> list[tuple[int, str]]:
    """返回该视频所有分P的 (cid, 分P标题)；单P返回 [(cid, 主标题)]"""
    url = "https://api.bilibili.com/x/web-interface/view"
    r = httpx.get(url, params={"bvid": bvid}, headers=_headers(), timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    pages = data.get("pages") or []
    if not pages:
        return [(data["cid"], data["title"])]
    return [(p["cid"], p.get("part") or data["title"]) for p in pages]


def _fetch_subtitle(bvid: str, cid: int, ai: bool = False) -> str | None:
    """抓取单分P字幕文本。ai=False 取 CC 字幕，ai=True 取 AI 字幕；无则返回 None"""
    from app.bilibili import _pick_subtitle_url

    url = "https://api.bilibili.com/x/player/wbi/v2"
    params = {"cid": cid, "bvid": bvid}
    r = httpx.get(url, params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    subs = r.json()["data"].get("subtitle", {}).get("subtitles", [])
    if not subs:
        return None
    if ai:
        ai_subs = [s for s in subs if s.get("lan", "").startswith("ai-")]
        if not ai_subs:
            return None
        sub_url = ai_subs[0]["subtitle_url"]
    else:
        cc_subs = [s for s in subs if not s.get("lan", "").startswith("ai-")]
        if not cc_subs:
            return None
        sub_url = _pick_subtitle_url(cc_subs)
    if sub_url.startswith("//"):
        sub_url = "https:" + sub_url  # B站返回协议相对URL
    body = httpx.get(sub_url, headers=_headers(), timeout=20).json().get("body", [])
    if not body:
        return None
    return "\n".join(f"[{int(item['from'])}s] {item['content']}" for item in body)


def three_tier_transcribe(bvid: str, cid: int) -> dict:
    """三级降级转录：CC字幕 -> AI字幕 -> 需ASR。返回 {source, text, tier}"""
    cc = _fetch_subtitle(bvid, cid, ai=False)
    if cc:
        return {"source": "cc", "text": cc, "tier": "cc"}
    ai = _fetch_subtitle(bvid, cid, ai=True)
    if ai:
        return {"source": "ai", "text": ai, "tier": "ai"}
    return {"source": "asr_needed", "text": "", "tier": "asr"}
