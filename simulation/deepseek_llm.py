"""LangChain wrapper for DeepSeek — overrides with_structured_output to use JSON parsing."""
import json
import re
from typing import Type
from pydantic import BaseModel, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


class _StructuredOutputWrapper:
    """Asks for JSON in the prompt and parses the response."""

    def __init__(self, llm, schema: Type[BaseModel]):
        self._llm    = llm
        self._schema = schema

    def invoke(self, messages: list, **kwargs) -> BaseModel:
        json_instruction = (
            f"\nRespond with a valid JSON object only. "
            f"Match this schema exactly: {self._schema.model_json_schema()}"
        )
        augmented = list(messages)
        last = augmented[-1]
        content = last["content"] if isinstance(last, dict) else last.content
        role    = last.get("role", "user") if isinstance(last, dict) else "user"
        augmented[-1] = {"role": role, "content": content + json_instruction}

        result = self._llm.invoke(augmented)
        from agents.base_agent import _count
        _count(result)
        text   = result.content if hasattr(result, "content") else str(result)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        data  = json.loads(match.group()) if match else {}
        return self._schema(**data)


class DeepSeekChatModel(ChatOpenAI):
    """ChatOpenAI pointed at DeepSeek with JSON-parsing structured output."""

    def with_structured_output(self, schema: Type[BaseModel], **kwargs) -> _StructuredOutputWrapper:
        return _StructuredOutputWrapper(self, schema)
