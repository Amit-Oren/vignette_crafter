"""LangChain-compatible wrapper for the lab Tailscale REST API."""
import json
import re
import requests
from typing import Any, List, Optional, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _StructuredOutputWrapper:
    """Mimics a structured LLM by injecting a JSON instruction and parsing the response."""

    def __init__(self, llm, schema: Type[BaseModel]):
        self._llm    = llm
        self._schema = schema

    def invoke(self, messages: list, **kwargs) -> BaseModel:
        # Append JSON instruction to the last user message
        augmented = list(messages)
        json_instruction = (
            f"\nRespond with a valid JSON object only. "
            f"Match this schema exactly: {self._schema.model_json_schema()}"
        )
        last = augmented[-1]
        augmented[-1] = {"role": last.get("role", "user") if isinstance(last, dict) else "user",
                         "content": (last["content"] if isinstance(last, dict) else last.content)
                                    + json_instruction}

        result = self._llm.invoke(augmented)
        from agents.base_agent import _count
        _count(result)
        text = result.content if hasattr(result, "content") else str(result)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        return self._schema(**data)


class OpenSourceChatModel(BaseChatModel):
    """Wraps the lab REST API as a LangChain ChatModel.

    Works as a drop-in replacement for ChatOpenAI / ChatAnthropic.
    """

    model:       str
    api_url:     str
    api_key:     str
    temperature: float = 0.7
    timeout:     int   = 600

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  **kwargs: Any) -> ChatResult:
        # Separate system message from the rest
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), None)
        prompt = "\n".join(m.content for m in messages if not isinstance(m, SystemMessage))

        response = requests.post(
            self.api_url,
            headers={"x-api-key": self.api_key},
            json={"model": self.model, "prompt": prompt, "system": system},
            timeout=self.timeout,
        )
        response.raise_for_status()
        text = response.json()["text"]

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema: Type[BaseModel], **kwargs) -> _StructuredOutputWrapper:
        return _StructuredOutputWrapper(self, schema)

    @property
    def _llm_type(self) -> str:
        return "open_source"
