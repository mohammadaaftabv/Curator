# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small OpenAI-compatible client for NVIDIA-hosted judge models."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pnc_tuning.io_utils import (
    atomic_write_json,
    ensure_within_work_root,
    lexical_absolute_path,
    load_json,
    stable_json_hash,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pnc_tuning.config import NvidiaConfig


@dataclass(frozen=True)
class ChatResult:
    """Normalized chat-completion result."""

    model: str
    content: str
    usage: dict[str, Any]
    metadata: dict[str, Any]
    cached: bool
    cache_key: str


class NvidiaClient:
    """NVIDIA endpoint client with request hashing and work-root-scoped caching."""

    def __init__(self, config: NvidiaConfig, *, work_root: str | Path, cache_dir: str | Path = "cache") -> None:
        self.config = config
        self.work_root = lexical_absolute_path(work_root)
        self.cache_dir = ensure_within_work_root(cache_dir, self.work_root)
        self._client: Any = None

    @property
    def api_key(self) -> str:
        """Read the API key lazily without logging it."""

        return os.environ.get(self.config.api_key_env, "")

    def list_models(self) -> set[str]:
        """Discover live model IDs from ``GET /v1/models``."""

        url = f"{self.config.base_url.rstrip('/')}/models"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
            payload = json.load(response)
        return {str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")}

    def _get_client(self) -> Any:  # noqa: ANN401
        if self._client is None:
            if not self.api_key:
                msg = f"Environment variable {self.config.api_key_env} is required for chat requests"
                raise RuntimeError(msg)
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.api_key,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
        return self._client

    def chat(  # noqa: PLR0913
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        json_output: bool = False,
        disable_thinking: bool = True,
    ) -> ChatResult:
        """Run or load one deterministic chat request."""

        request_payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
        }
        if json_output:
            request_payload["response_format"] = {"type": "json_object"}
        if disable_thinking:
            request_payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        cache_key = stable_json_hash(
            {
                "endpoint": self.config.base_url,
                "request": request_payload,
                "cache_schema": 1,
            }
        )
        cache_path = ensure_within_work_root(self.cache_dir / f"{cache_key}.json", self.work_root)
        if cache_path.is_file():
            cached = load_json(cache_path)
            return ChatResult(
                model=str(cached["model"]),
                content=str(cached["content"]),
                usage=dict(cached.get("usage", {})),
                metadata=dict(cached.get("metadata", {})),
                cached=True,
                cache_key=cache_key,
            )

        response = self._get_client().chat.completions.create(**request_payload)
        choice = response.choices[0] if response.choices else None
        content = "" if choice is None or choice.message.content is None else str(choice.message.content)
        usage = response.usage.model_dump() if response.usage is not None else {}
        metadata = {
            "created": getattr(response, "created", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": None if choice is None else getattr(choice, "finish_reason", None),
        }
        value = {
            "model": response.model or model,
            "content": content,
            "usage": usage,
            "metadata": metadata,
            "cache_key": cache_key,
        }
        atomic_write_json(cache_path, value, self.work_root)
        return ChatResult(
            model=str(value["model"]),
            content=content,
            usage=usage,
            metadata=metadata,
            cached=False,
            cache_key=cache_key,
        )
