from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from config import Settings, get_env


@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.api_key = get_env(self.settings.api_key)
        self.base_url = get_env(self.settings.base_url)
        self.model = get_env(self.settings.model_name)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        ).with_options(timeout=self.settings.request_timeout_sec)

    def chat_completion(self, messages: list[dict[str, str]]) -> LLMResponse:
        # messages format:
        # [
        #     {"role": "system", "content": "..."},
        #     {"role": "user", "content": "..."},
        # ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
        )
        content = completion.choices[0].message.content or ""
        raw = completion.model_dump() if hasattr(completion, "model_dump") else {}
        return LLMResponse(content=content, raw=raw)
