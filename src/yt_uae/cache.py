from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import aiosqlite


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS kv_cache (
  cache_key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS processed (
  item_type TEXT NOT NULL,
  item_id TEXT NOT NULL,
  status TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (item_type, item_id)
);
"""


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0


class SqliteCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self.stats = CacheStats()

    async def open(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def get_json(self, key: str, *, max_age_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
        assert self._db is not None
        row = await (await self._db.execute(
            "SELECT value_json, updated_at FROM kv_cache WHERE cache_key = ?",
            (key,),
        )).fetchone()
        if not row:
            self.stats = CacheStats(self.stats.hits, self.stats.misses + 1, self.stats.writes)
            return None
        value_json, updated_at = row
        if max_age_seconds is not None:
            if int(time.time()) - int(updated_at) > max_age_seconds:
                self.stats = CacheStats(self.stats.hits, self.stats.misses + 1, self.stats.writes)
                return None
        self.stats = CacheStats(self.stats.hits + 1, self.stats.misses, self.stats.writes)
        return json.loads(value_json)

    async def set_json(self, key: str, value: Dict[str, Any]) -> None:
        assert self._db is not None
        now = int(time.time())
        await self._db.execute(
            "INSERT INTO kv_cache(cache_key, value_json, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(cache_key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False), now),
        )
        await self._db.commit()
        self.stats = CacheStats(self.stats.hits, self.stats.misses, self.stats.writes + 1)

    async def mark_processed(self, item_type: str, item_id: str, status: str) -> None:
        assert self._db is not None
        now = int(time.time())
        await self._db.execute(
            "INSERT INTO processed(item_type, item_id, status, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(item_type,item_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
            (item_type, item_id, status, now),
        )
        await self._db.commit()

    async def is_processed(self, item_type: str, item_id: str) -> bool:
        assert self._db is not None
        row = await (await self._db.execute(
            "SELECT 1 FROM processed WHERE item_type = ? AND item_id = ?",
            (item_type, item_id),
        )).fetchone()
        return row is not None

