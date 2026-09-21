"""What a task is: prompts, tools and the report it produces."""
import hashlib
from dataclasses import dataclass

from reports import BaseReport


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    system_prompt: str
    report_prompt: str  # asks for the final report; names the fields of report_model
    report_model: type[BaseReport]
    user_prompt: str = "{ticker}"  # formatted with the subject of the run
    tools: tuple[str, ...] | None = None  # names from the tool registry; None = all tools

    @property
    def prompt_hash(self) -> str:
        """Short fingerprint of the prompts, so stored runs show which version produced them."""
        text = "\n".join([self.system_prompt, self.user_prompt, self.report_prompt])
        return hashlib.sha256(text.encode()).hexdigest()[:8]
