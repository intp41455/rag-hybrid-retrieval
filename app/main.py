import uuid
from fastapi import FastAPI, Header, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.config import settings
from app.models import Entry, Todo
from app.bilibili import fetch_bilibili_entry, NoSubtitleError, BiliApiError
from app.upload_doc import parse_uploaded_doc_entry
from app.upload_media import transcribe_media_entry
from app.web_fetch import fetch_web_article
from app.corrector import correct_entry
from app.embed_store import chunk_and_store_entry
from app.analyzer import analyze_entry
from app.visualizer import visualize_entry
from app.tts import split_text_for_tts, synthesize
from app.query_engine import answer_query
from app.store import EntryStore
from app.todo_store import TodoStore
from app.bilibili_fav import list_favorites
from app.jobs import JobQueue, enqueue_fav_sync
from app.watchdir import scan_watchdir
from app.knowledge_graph import KnowledgeGraph

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _auth(auth: str | None):
    if not auth or auth.replace("Bearer ", "") != settings.bearer_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


class BiliReq(BaseModel):
    bvid: str
    language: str = "auto"
    page: int | None = None  # None=采集全部分P；n=只采集第 n 个分P


class WebReq(BaseModel):
    url: str
    language: str = "auto"


class NoteReq(BaseModel):
    title: str
    content: str
    tags: list[str] = []


class QueryReq(BaseModel):
    query: str
    language: str = "auto"


class TodoReq(BaseModel):
    content: str
    tags: list[str] = []
    entry_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/entries")
def list_entries(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return EntryStore().list_entries()


@app.get("/entries/{entry_id}")
def get_entry(entry_id: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = EntryStore().get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    return entry


@app.post("/ingest/bilibili")
def ingest_bilibili(req: BiliReq, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    try:
        entry = fetch_bilibili_entry(req.bvid, page=req.page)
    except NoSubtitleError as e:
        return {"status": "no_subtitle", "message": str(e)}
    except BiliApiError as e:
        return {"status": "api_error", "message": str(e)}
    return _process_entry(entry, req.language)


@app.post("/ingest/upload-doc")
async def ingest_upload_doc(language: str = "auto", file: UploadFile = File(...), auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = await parse_uploaded_doc_entry(file)
    if entry.status == "encoding_error":
        return {"status": "encoding_error", "message": entry.error_message, "entry": entry.model_dump()}
    return _process_entry(entry, language)


@app.post("/ingest/upload-media")
async def ingest_upload_media(language: str = "auto", file: UploadFile = File(...), auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = await transcribe_media_entry(file, language)
    return _process_entry(entry, language)


@app.post("/ingest/web")
def ingest_web(req: WebReq, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = fetch_web_article(req.url)
    if entry.status == "fetch_error":
        return {"status": "fetch_error", "message": entry.error_message, "entry": entry.model_dump()}
    return _process_entry(entry, req.language)


@app.post("/ingest/note")
def ingest_note(req: NoteReq, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = Entry(
        id=str(uuid.uuid4()), type="note", source="note", title=req.title,
        raw_text=req.content, corrected_text=req.content, tags=req.tags, status="ok"
    )
    return _process_entry(entry, "auto")


@app.post("/ingest/manual")
def ingest_manual(entry: Entry, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return _process_entry(entry, entry.language or "auto")


def _process_entry(entry: Entry, language: str):
    entry = correct_entry(entry, language)
    chunk_and_store_entry(entry)
    entry.analysis = analyze_entry(entry)
    entry.visualization = visualize_entry(entry)
    EntryStore().save_entry(entry)
    # 同步更新知识图谱/知识树
    try:
        KnowledgeGraph().add_entry(entry.id, entry.corrected_text)
    except Exception:
        pass
    return {"status": "ok", "entry": entry.model_dump()}


@app.post("/query")
def query(req: QueryReq, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return answer_query(req.query, req.language)


@app.post("/tts/chunks")
def tts_chunks(entry_id: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    entry = EntryStore().get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    chunks = split_text_for_tts(entry.corrected_text)
    return {"chunks": chunks}


@app.post("/tts/speak")
def tts_speak(text: str, language: str = "zh", auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    audio = synthesize(text, language)
    from fastapi.responses import Response
    return Response(content=audio, media_type="audio/wav")


@app.get("/todos")
def list_todos(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return TodoStore().list()


@app.post("/todos")
def create_todo(req: TodoReq, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return TodoStore().create(req.content, tags=req.tags, entry_id=req.entry_id)


@app.post("/todos/{todo_id}/toggle")
def toggle_todo(todo_id: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    todo = TodoStore().toggle(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Not found")
    return todo


@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    if not TodoStore().delete(todo_id):
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---- B站收藏夹批量采集 ----
@app.get("/collections")
def collections(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return list_favorites()


@app.post("/collections/{fav_id}/sync")
def sync_collection(fav_id: int, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    job = enqueue_fav_sync(fav_id)
    return {"status": "ok", "job": job.model_dump()}


# ---- 任务队列 & 网盘目录监控 ----
@app.get("/jobs")
def jobs(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return JobQueue().list()


@app.post("/watchdir/scan")
def watchdir_scan(path: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    files = scan_watchdir(path)
    return {"files": files, "count": len(files)}


# ---- 知识图谱 / 知识树 ----
@app.get("/knowledge/nodes")
def knowledge_nodes(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return KnowledgeGraph().list_nodes()


@app.get("/knowledge/edges")
def knowledge_edges(auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return KnowledgeGraph().list_edges()


@app.get("/knowledge/tree")
def knowledge_tree(name: str, auth: str = Header(None, alias="Authorization")):
    _auth(auth)
    return KnowledgeGraph().build_tree(name)


# 静态托管 web 前端：API 路由优先，未匹配路径落到 web 目录
# 访问 http://<host>/ 即得 index.html，/app.js、/styles.css 同源可加载
app.mount("/", StaticFiles(directory="web", html=True), name="web")
