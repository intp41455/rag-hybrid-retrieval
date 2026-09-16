import json
import os
import uuid
from app.config import settings
from app.models import Job


class JobQueue:
    def __init__(self, data_dir: str | None = None):
        self.path = os.path.join(data_dir or settings.data_dir, "jobs.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.jobs = self._load()

    def _load(self) -> list[Job]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [Job(**item) for item in json.load(f)]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([j.model_dump() for j in self.jobs], f, ensure_ascii=False, default=str, indent=2)

    def create(self, kind: str, collection_id: str = "", total: int = 0) -> Job:
        job = Job(id=str(uuid.uuid4()), collection_id=collection_id, kind=kind,
                  status="queued", progress=0, total=total, processed=0, failed=[])
        self.jobs.append(job)
        self.save()
        return job

    def get(self, job_id: str) -> Job | None:
        for j in self.jobs:
            if j.id == job_id:
                return j
        return None

    def list(self) -> list[Job]:
        return sorted(self.jobs, key=lambda j: j.created_at, reverse=True)

    def update_progress(self, job_id: str, processed: int):
        j = self.get(job_id)
        if not j:
            return
        j.processed = processed
        j.progress = int(processed / j.total * 100) if j.total else 100
        j.status = "done" if j.progress >= 100 else "running"
        self.save()

    def mark_failed(self, job_id: str, item: str):
        j = self.get(job_id)
        if not j:
            return
        j.failed.append(item)
        self.save()


def enqueue_fav_sync(fav_id: int) -> Job:
    """创建收藏夹批量采集任务。真实后台线程在部署环境执行，这里登记任务即返回。"""
    from app.bilibili_fav import list_fav_videos
    q = JobQueue()
    videos = list_fav_videos(fav_id)
    job = q.create("fav_sync", collection_id=str(fav_id), total=len(videos))
    return job
