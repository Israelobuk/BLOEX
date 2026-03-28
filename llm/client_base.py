from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class LLMClient(ABC):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        backend_name: str = "",
        provider_label: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.backend_name = backend_name or self.__class__.__name__.lower()
        self.provider_label = provider_label or self.__class__.__name__

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "backend": self.backend_name,
            "client": self.provider_label,
        }
