"""Task registry. To add a task: create a module with a Task and list it here."""

from .base import Task
from .generic import GENERIC
from .sentiment import SENTIMENT

TASKS: dict[str, Task] = {t.name: t for t in (SENTIMENT, GENERIC)}

__all__ = ["TASKS", "Task"]
