from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse: ...


class OpenAICompatibleModel:
    """Minimal Chat Completions client; no vendor SDK or agent framework."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        self.reasoning_effort = reasoning_effort or os.environ.get("OPENAI_REASONING_EFFORT", "")
        timeout_value = timeout_seconds if timeout_seconds is not None else os.environ.get("OPENAI_TIMEOUT_SECONDS", "120")
        try:
            self.timeout_seconds = float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be a positive number") from exc
        if self.timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be a positive number")
        if not all((self.base_url, self.api_key, self.model)):
            raise ValueError("OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL are required")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        payload_data = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        if self.reasoning_effort:
            payload_data["reasoning_effort"] = self.reasoning_effort
        payload = json.dumps(payload_data).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model HTTP error {exc.code}: {body[:1000]}") from exc
        message = data["choices"][0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = {"__invalid_json__": arguments}
            calls.append({"id": raw.get("id"), "name": function.get("name"), "arguments": arguments})
        return ModelResponse(message.get("content"), calls)


class ScriptedModel:
    """Deterministic model used by tests and demonstrations."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
        self.requests.append((messages, tools))
        if not self.responses:
            raise RuntimeError("ScriptedModel has no response left")
        return self.responses.pop(0)
