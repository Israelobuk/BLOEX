from __future__ import annotations

import os
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE_PATH = BASE_DIR / "app.db"


def _sqlite_path() -> Path:
    raw = os.getenv("SQLITE_DB_PATH", "").strip()
    if not raw:
        return DEFAULT_SQLITE_PATH
    path = Path(raw)
    if not path.is_absolute():
        return (BASE_DIR / path).resolve()
    return path


def _db_url() -> str:
    path = _sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


engine = create_engine(
    _db_url(),
    connect_args={"check_same_thread": False},
    echo=False,
)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
