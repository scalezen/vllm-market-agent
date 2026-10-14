"""Task registry. To add a task: create a module with a Task and list it here."""
from .base import Task
from .sentiment import SENTIMENT
from .generic import GENERIC

TASKS: dict[str, Task] = {t.name: t for t in (SENTIMENT, GENERIC)}

__all__ = ["TASKS", "Task"]
