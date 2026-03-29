from typing import Dict, List

import requests

from .client_base import LLMClient


class OllamaClient(LLMClient):
    """
    Ollama API docs:
    https://docs.ollama.com/api
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        backend_name: str = "ollama",
        provider_label: str = "Ollama",
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            backend_name=backend_name,
            provider_label=provider_label,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def healthcheck(self):
        if not self.base_url:
            return False, "Model service URL is required."
        if not self.model:
            return False, "Model name is required."

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._headers(),
                timeout=min(max(self.timeout_seconds, 5), 30),
            )
            response.raise_for_status()
            data = response.json()
            models = [item.get("name", "") for item in data.get("models", []) if isinstance(item, dict)]
            if models and self.model not in models:
                return False, f"Connected to Ollama, but model '{self.model}' is not available on that server."
            return True, f"Connected to {self.provider_label}. Model configured: {self.model}"
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 401:
                return False, f"{self.provider_label} rejected the request. Check the API key for {self.base_url}."
            return False, f"Cannot connect to {self.provider_label} at {self.base_url}. {exc}"
        except requests.RequestException as exc:
            return False, f"Cannot connect to {self.provider_label} at {self.base_url}. {exc}"

    def _post_generate(
        self,
        url: str,
        payload: Dict[str, object],
        timeout_seconds: int | None = None,
    ) -> Dict[str, object]:
        response = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=timeout_seconds or self.timeout_seconds,
        )
        if not response.ok:
            body = response.text.strip()
            detail = body[:500] if body else f"HTTP {response.status_code}"
            raise RuntimeError(f"Ollama generate failed ({response.status_code}): {detail}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Ollama response format: {data}")
        return data

    def _extract_generate_content(self, data: Dict[str, object]) -> str:
        try:
            return str(data["response"])
        except KeyError as exc:
            raise RuntimeError(f"Unexpected Ollama response format: {data}") from exc

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        blocks: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper() or "USER"
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            blocks.append(f"{role}:\n{content}")
        blocks.append("ASSISTANT:")
        return "\n\n".join(blocks)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int | None = None,
    ) -> str:
        msg_text = "\n".join(str(m.get("content", "")) for m in messages)
        wants_json = (
            "OUTPUT JSON SCHEMA" in msg_text
            or "STRICT JSON" in msg_text
            or "Return this exact JSON object shape" in msg_text
        )
        generate_url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": self._messages_to_prompt(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if wants_json:
            payload["format"] = "json"
        data = self._post_generate(generate_url, payload, timeout_seconds=timeout_seconds)
        return self._extract_generate_content(data)
