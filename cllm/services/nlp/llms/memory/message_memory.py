from typing import List, Optional, Dict
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from .utils import count_tokens, get_max_context_length


class MessageMemory:
    def __init__(
        self,
        max_tokens: int = -1,
        margin: int = 1500,
        messages: Optional[List[BaseMessage]] = None,
    ) -> None:
        self.max_tokens = max_tokens if max_tokens > 0 else 8e8
        self.margin = margin
        self.init_messages(messages)

    def reset(self) -> List[BaseMessage]:
        self.init_messages()
        return self.stored_messages

    def init_messages(self, messages=None) -> None:
        if messages is not None:
            self.stored_messages = messages
        else:
            self.stored_messages = []

    @classmethod
    def to_messages(cls, items: List[Dict]):
        messages = []
        for m in items:
            if (
                not isinstance(m, dict)
                or m.get("role", None) is None
                or m.get("role") not in ["user", "assistant", "system"]
            ):
                raise TypeError()

            if m["role"] == "system":
                messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))

        return messages

    def to_dict(self):
        messages = []
        for m in self.stored_messages:
            if not isinstance(m, BaseMessage) or m.type is None:
                raise TypeError()

            if isinstance(m, SystemMessage):
                messages.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                messages.append({"role": "assistant", "content": m.content})

        return messages

    def get_memory(self):
        return self.stored_messages

    def update_message(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def insert_messages(
        self, idx: int = 0, messages: List[BaseMessage] = None
    ) -> List[BaseMessage]:
        for m in messages[::-1]:
            self.stored_messages.insert(idx, m)
        return self.stored_messages

    @classmethod
    def messages2str(self, history):
        history_text = ""
        for m in history:
            if isinstance(m, SystemMessage):
                history_text += "<system>: " + m.content + "\n"
            elif isinstance(m, HumanMessage):
                history_text += "<user>: " + m.content + "\n"
            elif isinstance(m, AIMessage):
                history_text += "<assistant>: " + m.content + "\n"
        return history_text

    def memory2str(self):
        return self.messages2str(self.stored_messages)

    def cut_memory(self, LLM_encoding: str):
        start = 0
        while start <= len(self.stored_messages):
            # print(f'self.stored_messages = {self.stored_messages}')
            history = self.stored_messages[start:]
            history_text = self.messages2str(history)
            num = count_tokens(LLM_encoding, history_text)
            max_tokens = min(self.max_tokens, get_max_context_length(LLM_encoding))
            if max_tokens - num > self.margin:
                self.stored_messages = self.stored_messages[start:]
                return self.stored_messages

            start += 1
        self.init_messages()
        return self.stored_messages


if __name__ == "__main__":
    import os

    os.environ["TIKTOKEN_CACHE_DIR"] = "/mnt/petrelfs/liuzhaoyang/workspace/tmp"
    messages = [
        SystemMessage(content="SystemMessage 1"),
        HumanMessage(content="Remember a = 5 * 4."),
        AIMessage(content="SystemMessage 2"),
        HumanMessage(content="what is the value of a?"),
    ] * 400
    print(SystemMessage(content="SystemMessage 1").content)
    print(len(messages))
    mem = MessageMemory(
        -1,
        messages,
    )
    messages = mem.cut_memory("gpt-3.5-turbo")
    print(len(messages))
