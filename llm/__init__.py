from urllib.parse import urlparse

from .client_ollama import OllamaClient


LOCAL_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0', 'host.docker.internal'}


def _provider_label(base_url: str) -> str:
    hostname = (urlparse(base_url).hostname or '').lower()
    return 'Ollama' if hostname in LOCAL_HOSTS else 'Ollama Server'


def create_client(
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_seconds: int = 120,
):
    return OllamaClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        backend_name='ollama',
        provider_label=_provider_label(base_url),
    )
