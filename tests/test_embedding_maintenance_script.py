import importlib.util
import sqlite3
from pathlib import Path


def _load_cleanup_module():
    path = Path("scripts/cleanup_orphan_embeddings.py")
    spec = importlib.util.spec_from_file_location("cleanup_orphan_embeddings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_orphan_embeddings_returns_rows_without_live_bucket():
    cleanup = _load_cleanup_module()

    rows = [("live", "2026-05-25"), ("gone", "2026-05-25")]

    assert cleanup.find_orphan_embeddings(rows, {"live"}) == [("gone", "2026-05-25")]


def test_delete_embeddings_removes_only_requested_ids(tmp_path):
    cleanup = _load_cleanup_module()
    db_path = tmp_path / "embeddings.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE embeddings (bucket_id TEXT PRIMARY KEY, embedding TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO embeddings (bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
        [("live", "[]", "2026-05-25"), ("gone", "[]", "2026-05-25")],
    )
    conn.commit()
    conn.close()

    deleted = cleanup.delete_embeddings(str(db_path), ["gone"])

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT bucket_id FROM embeddings ORDER BY bucket_id").fetchall()
    conn.close()
    assert deleted == 1
    assert rows == [("live",)]
