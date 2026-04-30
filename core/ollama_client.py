from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass(frozen=True)
class OllamaResult:
    ok: bool
    text: str
    error: str = ""


class AsyncOllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 60,
        api_key: str = "",
        embedding_model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self.embedding_model = embedding_model or model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health(self) -> dict[str, Any]:
        if not self.base_url:
            return {"ok": False, "status": "Ollama URL is not configured."}
        try:
            timeout = aiohttp.ClientTimeout(total=min(max(self.timeout_seconds, 3), 15))
            async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return {"ok": 200 <= response.status < 400, "status": f"HTTP {response.status}"}
        except Exception as exc:
            return {"ok": False, "status": str(exc)}

    async def generate(self, prompt: str, *, json_mode: bool = False) -> OllamaResult:
        if not self.base_url or not self.model:
            return OllamaResult(ok=False, text="", error="Ollama is not configured.")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if not 200 <= response.status < 400:
                        return OllamaResult(ok=False, text="", error=f"HTTP {response.status}")
                    data = await response.json()
                    return OllamaResult(ok=True, text=str(data.get("response", "")).strip())
        except Exception as exc:
            return OllamaResult(ok=False, text="", error=str(exc))

    async def embedding(self, text: str) -> list[float] | None:
        if not self.base_url or not self.embedding_model:
            return None
        payload = {"model": self.embedding_model, "prompt": text}
        try:
            timeout = aiohttp.ClientTimeout(total=min(max(self.timeout_seconds, 5), 30))
            async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
                async with session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                    if not 200 <= response.status < 400:
                        return None
                    data = await response.json()
                    embedding = data.get("embedding")
                    if isinstance(embedding, list):
                        return [float(value) for value in embedding]
        except Exception:
            return None
        return None
