from typing import Dict, List

import requests

from .client_base import LLMClient


MAX_CONTINUATION_PASSES = 2


class OllamaClient(LLMClient):
    """
    Ollama API docs:
    https://docs.ollama.com/api
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        backend_name: str = "ollama",
        provider_label: str = "Ollama",
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            backend_name=backend_name,
            provider_label=provider_label,
        )

    def healthcheck(self):
        if not self.base_url:
            return False, "Model service URL is required."
        if not self.model:
            return False, "Model name is required."

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(max(self.timeout_seconds, 5), 30),
            )
            response.raise_for_status()
            data = response.json()
            models = [item.get("name", "") for item in data.get("models", []) if isinstance(item, dict)]
            if models and self.model not in models:
                return False, f"Connected to Ollama, but model '{self.model}' is not available on that server."
            return True, f"Connected to {self.provider_label}. Model configured: {self.model}"
        except requests.RequestException as exc:
            return False, f"Cannot connect to {self.provider_label} at {self.base_url}. {exc}"

    def _post_chat(
        self,
        url: str,
        payload: Dict[str, object],
        timeout_seconds: int | None = None,
    ) -> Dict[str, object]:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout_seconds or self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Ollama response format: {data}")
        return data

    def _extract_content(self, data: Dict[str, object]) -> str:
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response format: {data}") from exc

    def _response_was_cut_off(self, data: Dict[str, object], content: str, max_tokens: int) -> bool:
        done_reason = str(data.get("done_reason", "")).strip().lower()
        if done_reason in {"length", "max_tokens"}:
            return True

        stripped = content.rstrip()
        if not stripped:
            return False

        if len(stripped.split()) < max(60, max_tokens // 3):
            return False

        return stripped.endswith(("-", ":", ",", ";", "(", "[", "{", "/"))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int | None = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        msg_text = "\n".join(str(m.get("content", "")) for m in messages)
        wants_json = (
            "OUTPUT JSON SCHEMA" in msg_text
            or "STRICT JSON" in msg_text
            or "Return this exact JSON object shape" in msg_text
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if wants_json:
            payload["format"] = "json"

        data = self._post_chat(url, payload, timeout_seconds=timeout_seconds)
        content = self._extract_content(data)

        if wants_json:
            return content

        current_messages = list(messages)
        current_content = content

        for _ in range(MAX_CONTINUATION_PASSES):
            if not self._response_was_cut_off(data, current_content, max_tokens):
                break

            current_messages = current_messages + [
                {"role": "assistant", "content": current_content},
                {
                    "role": "user",
                    "content": "Continue exactly from where you stopped. Do not restart, summarize, or repeat earlier text.",
                },
            ]
            continuation_payload = {
                "model": self.model,
                "messages": current_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            data = self._post_chat(url, continuation_payload, timeout_seconds=timeout_seconds)
            continuation = self._extract_content(data).lstrip()
            if not continuation:
                break
            current_content = f"{current_content.rstrip()} {continuation}".strip()

        return current_content
