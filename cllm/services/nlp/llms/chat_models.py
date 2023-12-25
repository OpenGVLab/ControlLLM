import os
from turtle import mode
import openai
import requests
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel

from cllm.services.nlp.llms.memory import MessageMemory
from cllm.utils import timeout


class ChatOpenAI:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        model_kwargs: Dict[str, Any] = dict(),
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.model_kwargs = model_kwargs
        self.api_key = os.environ.get("OPENAI_API_KEY", openai_api_key)
        self.base_url = os.environ.get("OPENAI_BASE_URL", openai_base_url)
        # openai.api_key = self.api_key
        # openai.base_url = self.base_url

    def __call__(self, messages: List[BaseMessage], **kwargs):
        stream = kwargs.get("stream", False)
        context = MessageMemory(messages=messages)
        context.cut_memory(self.model_name)
        response = self.send_message(messages=context.to_dict(), stream=stream)
        return response

    def get_response(self, response):
        return response.choices[0].message.content

    def send_message(self, messages, stream=False):
        cnt = 10
        while cnt > 0:
            try:
                result = self.get_response(
                    self._send_message(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        stream=stream,
                        timeout=5,
                    )
                )
                break
            except Exception as e:
                cnt -= 1
                print(e)
                result = e
        return result

    # @timeout(5)
    def _send_message(self, *args, **kwargs):
        # return self.client.chat.completions.create(*args, **kwargs)
        # return openai.Completion.create(*args, **kwargs)
        return openai.chat.completions.create(*args, **kwargs)


class ChatLLAMA2(SimpleChatModel):
    """Wrapper around LLAMA2

    To use, you should launch you local model as web services.
    """

    client: Any = None  #: :meta private:
    endpoint: str = "http://localhost:10051"

    HUMAN_PROMPT = "user"
    AI_PROMPT = "assistant"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "local-chat"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        data = self._convert_messages_to_prompt(messages)
        response = requests.post(self.endpoint, json=data)
        return response.content.decode()

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = {
                "role": message.role.capitalize(),
                "content": message.content,
            }
        elif isinstance(message, HumanMessage):
            message_text = {"role": self.HUMAN_PROMPT, "content": message.content}
        elif isinstance(message, AIMessage):
            message_text = {"role": self.AI_PROMPT, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_text = {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return [self._convert_one_message_to_text(message) for message in messages]

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        return self._convert_messages_to_text(messages)


class ChatLLAMA2(SimpleChatModel):
    """Wrapper around LLAMA2

    To use, you should launch you local model as web services.
    """

    client: Any = None  #: :meta private:
    endpoint: str = "http://localhost:10051"

    HUMAN_PROMPT = "user"
    AI_PROMPT = "assistant"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "local-chat"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        data = self._convert_messages_to_prompt(messages)
        response = requests.post(self.endpoint, json=data)
        return response.content.decode()

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = {
                "role": message.role.capitalize(),
                "content": message.content,
            }
        elif isinstance(message, HumanMessage):
            message_text = {"role": self.HUMAN_PROMPT, "content": message.content}
        elif isinstance(message, AIMessage):
            message_text = {"role": self.AI_PROMPT, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_text = {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return [self._convert_one_message_to_text(message) for message in messages]

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        return self._convert_messages_to_text(messages)


if __name__ == "__main__":
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    msg = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
    ]
    response = chat(msg)
    print(response)
