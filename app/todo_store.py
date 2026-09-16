import json
import os
import uuid
from app.config import settings
from app.models import Todo


class TodoStore:
    def __init__(self, data_dir: str | None = None):
        self.path = os.path.join(data_dir or settings.data_dir, "todos.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.todos = self._load()

    def _load(self) -> list[Todo]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [Todo(**item) for item in json.load(f)]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([t.model_dump() for t in self.todos], f, ensure_ascii=False, default=str, indent=2)

    def create(self, content: str, tags: list[str] = None, entry_id: str = None) -> Todo:
        todo = Todo(id=str(uuid.uuid4()), content=content, tags=tags or [], entry_id=entry_id)
        self.todos.append(todo)
        self.save()
        return todo

    def list(self) -> list[Todo]:
        return sorted(self.todos, key=lambda t: t.created_at, reverse=True)

    def toggle(self, todo_id: str) -> Todo | None:
        for t in self.todos:
            if t.id == todo_id:
                t.done = not t.done
                self.save()
                return t
        return None

    def delete(self, todo_id: str) -> bool:
        before = len(self.todos)
        self.todos = [t for t in self.todos if t.id != todo_id]
        if len(self.todos) < before:
            self.save()
            return True
        return False
