from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken


def encryption_enabled() -> bool:
    raw = os.getenv("ENABLE_ENCRYPTION_UTILS", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_key() -> str:
    key = os.getenv("APP_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("APP_ENCRYPTION_KEY is required when encryption utilities are enabled.")
    return key


def validate_encryption_setup() -> None:
    if not encryption_enabled():
        return
    key = _get_key()
    Fernet(key.encode("utf-8"))


class FieldEncryptor:
    def __init__(self, key: str):
        self._fernet = Fernet(key.encode("utf-8"))

    def encrypt_text(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt_text(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Invalid encrypted payload.") from exc


def get_encryptor() -> FieldEncryptor:
    return FieldEncryptor(_get_key())
