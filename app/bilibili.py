import time
import uuid
import httpx
from app.config import settings
from app.models import Entry, Line

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"


class NoSubtitleError(Exception):
    pass


class BiliApiError(Exception):
    """B站接口请求失败（网络受限 / 视频不存在等）"""
    pass


def _bili_headers() -> dict:
    headers = {"User-Agent": _UA, "Referer": "https://www.bilibili.com"}
    if settings.bilibili_sessdata:
        headers["Cookie"] = f"SESSDATA={settings.bilibili_sessdata}"
    return headers


def _get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """带重试的 B站 GET 请求（网络偶发抖动时自动重试）。"""
    headers = _bili_headers()
    last_err = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1 + attempt)  # 1s, 2s 退避
    raise BiliApiError(f"B站接口请求失败（视频可能不存在或网络受限）: {last_err}") from last_err


def _bvid_info(bvid: str) -> tuple[int, str, list[dict]]:
    """返回 (首个cid, 标题, 分P列表)。
    分P列表元素: {"cid": int, "part": str}，单P视频时列表长度为 1。"""
    data = _get_json("https://api.bilibili.com/x/web-interface/view", params={"bvid": bvid})["data"]
    pages = data.get("pages") or []
    parts = [{"cid": p["cid"], "part": p.get("part", f"P{p.get('page', i+1)}")} for i, p in enumerate(pages)]
    if not parts:
        parts = [{"cid": data["cid"], "part": data["title"]}]
    return data["cid"], data["title"], parts


def _pick_subtitle_url(subs: list) -> str:
    priority = ["zh-CN", "zh-Hans", "zh", "ai-zh", "zh-TW", "en-US", "en"]
    for lan in priority:
        for s in subs:
            if s.get("lan", "").startswith(lan):
                return s["subtitle_url"]
    return subs[0]["subtitle_url"]


def _fetch_part_subtitles(bvid: str, cid: int) -> list[Line] | None:
    """抓取单个分P的 CC 字幕，无字幕返回 None。"""
    subs = _get_json("https://api.bilibili.com/x/player/wbi/v2", params={"cid": cid, "bvid": bvid})["data"].get("subtitle", {}).get("subtitles", [])
    if not subs:
        return None
    sub_url = _pick_subtitle_url(subs)
    if sub_url.startswith("//"):
        sub_url = "https:" + sub_url  # B站返回协议相对URL
    body = _get_json(sub_url).get("body", [])
    return [Line(start=int(item["from"]), end=int(item["to"]), text=item["content"]) for item in body]


def fetch_bilibili_entry(bvid: str, page: int | None = None) -> Entry:
    """采集 B 站视频 CC 字幕。
    page=None 采集全部分P（合并为一条）；page=n 只采集第 n 个分P。"""
    _, title, parts = _bvid_info(bvid)
    if page is not None:
        if not (1 <= page <= len(parts)):
            raise NoSubtitleError(f"分P不存在：共 {len(parts)}P，请求 P{page}")
        parts = [parts[page - 1]]

    all_lines: list[Line] = []
    missing: list[str] = []
    multi = len(parts) > 1
    for idx, part in enumerate(parts, start=1):
        lines = _fetch_part_subtitles(bvid, part["cid"])
        if lines is None:
            missing.append(part["part"])
            continue
        if multi:
            # 分P边界标记（以 [Pn] 形式嵌入，便于检索时定位）
            all_lines.append(Line(start=0, end=0, text=f"===== P{idx} {part['part']} ====="))
        all_lines.extend(lines)

    if not all_lines and missing:
        raise NoSubtitleError("该视频无 CC 字幕")
    raw_text = "\n".join(f"[{l.start}s] {l.text}" for l in all_lines)
    display_title = title if not multi or page else f"{title}（全 {len(parts)}P 合并）"
    return Entry(
        id=str(uuid.uuid4()),
        type="bilibili",
        source=bvid,
        title=display_title,
        raw_text=raw_text,
        corrected_text=raw_text,
        status="ok",
    )
