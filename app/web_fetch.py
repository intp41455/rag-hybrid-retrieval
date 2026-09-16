import uuid
import httpx
from bs4 import BeautifulSoup
from app.models import Entry


class FetchError(Exception):
    pass


SITE_SELECTORS = {
    "zhihu.com": ("article", {"class": "Post-RichTextContainer"}),
    "xiaohongshu.com": ("div", {"id": "detail-desc"}),
    "mp.weixin.qq.com": ("div", {"class": "rich_media_content"}),
}


def fetch_web_article(url: str) -> Entry:
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return Entry(
            id=str(uuid.uuid4()), type="web_article", source=url, title=url,
            raw_text="", corrected_text="", status="fetch_error",
            error_message=f"无法抓取页面：{e}"
        )
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text = ""
    for domain, (tag, attrs) in SITE_SELECTORS.items():
        if domain in url:
            el = soup.find(tag, attrs)
            if el:
                text = el.get_text("\n", strip=True)
                break
    if not text:
        article = soup.find("article") or soup.find("main") or soup.find("body")
        if article:
            text = article.get_text("\n", strip=True)
    if len(text) < 50:
        return Entry(
            id=str(uuid.uuid4()), type="web_article", source=url, title=title,
            raw_text="", corrected_text="", status="fetch_error",
            error_message="页面正文过短，请复制全文粘贴"
        )
    return Entry(
        id=str(uuid.uuid4()), type="web_article", source=url, title=title,
        raw_text=text, corrected_text=text, status="ok"
    )
