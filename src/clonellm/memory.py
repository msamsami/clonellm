from typing import Sequence, cast

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

__all__ = ("InMemoryHistory", "get_session_history", "get_session_history_size", "clear_session_history")


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)
    max_memory_size: int = Field(
        default=-1, description="Maximum number of messages in memory. -1 means no limit and 0 means no memory at all.", ge=-1
    )

    @staticmethod
    def _trim_messages(messages: Sequence[BaseMessage], max_memory_size: int) -> Sequence[BaseMessage]:
        if max_memory_size == 0:
            return []
        elif max_memory_size > 0:
            return messages[-max_memory_size:]
        else:
            return messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages to the store

        Args:
            messages (Sequence[BaseMessage]): A list of BaseMessage objects to store.
        """
        for message in messages:
            self.messages.append(message)
        self.messages = cast(list[BaseMessage], self._trim_messages(self.messages, self.max_memory_size))

    @root_validator
    def trim_messages_upon_init(cls, values):
        values["messages"] = cls._trim_messages(values["messages"], values["max_memory_size"])
        return values

    @property
    def memory_size(self) -> int:
        return len(self.messages)

    def clear(self) -> None:
        self.messages = []


_store: dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str, max_memory_size: int = -1) -> InMemoryHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryHistory(max_memory_size=max_memory_size)
    return _store[session_id]


def get_session_history_size(session_id: str) -> int:
    if session_id not in _store:
        return 0
    return len(_store[session_id].messages)


def clear_session_history(session_id: str) -> None:
    _ = _store.pop(session_id, None)
