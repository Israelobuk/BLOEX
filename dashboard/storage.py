from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoRunStore:
    def __init__(self, uri: str, database_name: str):
        self.uri = uri
        self.database_name = database_name
        self.enabled = bool(uri)

    def _collection(self):
        client = MongoClient(self.uri, serverSelectionTimeoutMS=2500)
        return client, client[self.database_name]["runs"]

    def save_run(self, payload: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False

        client = None
        try:
            client, collection = self._collection()
            document = dict(payload)
            document["created_at"] = datetime.now(timezone.utc)
            collection.insert_one(document)
            return True
        except PyMongoError:
            return False
        finally:
            if client is not None:
                client.close()

    def recent_runs(self, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        client = None
        try:
            client, collection = self._collection()
            cursor = collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        except PyMongoError:
            return []
        finally:
            if client is not None:
                client.close()
