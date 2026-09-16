from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class Line(BaseModel):
    start: int
    end: int
    text: str


class Chunk(BaseModel):
    entry_id: str
    text: str
    start: Optional[int] = None
    embedding: list[float] = []


class Analysis(BaseModel):
    summary: str = ""
    knowledge: list[str] = []
    expansion: list[str] = []


class Visualization(BaseModel):
    mindmap: str = ""
    flowchart: str = ""


class Entry(BaseModel):
    id: str
    type: str  # bilibili, netdisk_doc, netdisk_media, web_article, note, todo
    source: str
    title: str
    language: str = "auto"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = []
    raw_text: str = ""
    corrected_text: str = ""
    chunks: list[Chunk] = []
    analysis: Optional[Analysis] = None
    visualization: Optional[Visualization] = None
    status: str = "ok"
    error_message: str = ""


class Todo(BaseModel):
    id: str
    content: str
    done: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    due_at: Optional[datetime] = None
    tags: list[str] = []
    entry_id: Optional[str] = None


class Job(BaseModel):
    id: str
    collection_id: str = ""
    kind: str = ""
    status: str = "queued"  # queued / running / done / failed
    progress: int = 0
    total: int = 0
    processed: int = 0
    failed: list[str] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeNode(BaseModel):
    id: str
    name: str
    aliases: list[str] = []
    entries: list[str] = []
    summary: str = ""
    parents: list[str] = []
    children: list[str] = []


class KnowledgeEdge(BaseModel):
    source: str
    target: str
    relation: str = "相关"
    weight: float = 1.0
