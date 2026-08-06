"""
Simple in-memory store keyed by a random result ID, bridging /predict ->
/result/<id> -> /result/<id>/pdf without writing anything to disk (medical
images stay in memory only) and without a database (prediction history is
a deferred future feature — this is deliberately not that).

Single-process only: fine for a dev server or a single-worker deployment.
If you later run multiple gunicorn workers, this dict won't be shared
across them — that's exactly the point where a real database (Redis, or
the deferred SQLite history feature) becomes necessary, not before.
"""

import time
import uuid

import config

_store = {}


def _evict_expired():
    now = time.time()
    expired = [k for k, v in _store.items() if now - v["created_at"] > config.RESULT_TTL_SECONDS]
    for k in expired:
        del _store[k]


def save_result(data: dict) -> str:
    _evict_expired()
    result_id = uuid.uuid4().hex
    _store[result_id] = {"created_at": time.time(), "data": data}
    return result_id


def get_result(result_id: str):
    _evict_expired()
    entry = _store.get(result_id)
    return entry["data"] if entry else None
