from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def auth_secret() -> str:
    secret = os.getenv("AUTH_JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("AUTH_JWT_SECRET must be set.")
    return secret


def auth_alg() -> str:
    return os.getenv("AUTH_JWT_ALG", "HS256").strip() or "HS256"


def auth_expire_minutes() -> int:
    raw = os.getenv("AUTH_TOKEN_EXPIRE_MINUTES", "120").strip()
    return max(1, int(raw))


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(*, subject: str) -> tuple[str, int]:
    expires_in = auth_expire_minutes() * 60
    expire = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    payload = {"sub": subject, "exp": expire}
    token = jwt.encode(payload, auth_secret(), algorithm=auth_alg())
    return token, expires_in


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, auth_secret(), algorithms=[auth_alg()])
    except JWTError as exc:
        raise ValueError("Invalid token.") from exc

    subject = payload.get("sub")
    if not subject:
        raise ValueError("Invalid token subject.")
    return str(subject)
