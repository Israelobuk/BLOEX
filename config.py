from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _load_dotenv_file() -> None:
    env_path = Path('.env')
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_file()


@dataclass(frozen=True)
class AppConfig:
    model: str
    base_url: str
    api_key: str = ''
    temperature: float = 0.1
    max_tokens: int = 640
    timeout_seconds: int = 60
    critique_pass: bool = False


DEFAULT_CONFIG = AppConfig(
    model='tinyllama:latest',
    base_url='http://127.0.0.1:11434',
    max_tokens=120,
)


def _get_setting(name: str, default: str, secrets: Optional[Mapping[str, Any]] = None) -> str:
    env_value = os.getenv(name)
    if env_value not in (None, ''):
        return env_value

    if secrets and name in secrets:
        value = secrets[name]
        if value is None:
            return default
        return str(value)

    return default


def _get_bool(name: str, default: bool, secrets: Optional[Mapping[str, Any]] = None) -> bool:
    raw = _get_setting(name, str(default).lower(), secrets)
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def load_from_env(secrets: Optional[Mapping[str, Any]] = None) -> AppConfig:
    cfg = DEFAULT_CONFIG
    return AppConfig(
        model=_get_setting('BBE_MODEL', cfg.model, secrets).strip(),
        base_url=_get_setting('BBE_BASE_URL', cfg.base_url, secrets).strip(),
        api_key=_get_setting('BBE_API_KEY', _get_setting('OLLAMA_API_KEY', cfg.api_key, secrets), secrets).strip(),
        temperature=float(_get_setting('BBE_TEMPERATURE', str(cfg.temperature), secrets)),
        max_tokens=int(_get_setting('BBE_MAX_TOKENS', str(cfg.max_tokens), secrets)),
        timeout_seconds=int(_get_setting('BBE_TIMEOUT_SECONDS', str(cfg.timeout_seconds), secrets)),
        critique_pass=_get_bool('BBE_CRITIQUE_PASS', cfg.critique_pass, secrets),
    )
