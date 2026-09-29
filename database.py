from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from youtube_analytics.config import DATABASE_PATH, LOGGER


def connection(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    return db


@contextmanager
def managed_connection(path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    db = connection(path)
    try:
        with db:
            yield db
    finally:
        db.close()


def persist_dataset(frame: pd.DataFrame, path: Path = DATABASE_PATH) -> None:
    columns = [column for column in frame.columns if column != "published_date"]
    persist = frame[columns].copy()
    persist["published_at"] = pd.to_datetime(persist["published_at"], errors="coerce").astype(str)
    with managed_connection(path) as db:
        persist.to_sql("videos", db, if_exists="replace", index=False, chunksize=5000)
        db.execute("CREATE INDEX IF NOT EXISTS idx_videos_published ON videos(published_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_videos_category ON videos(category_name)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel_title)")
    LOGGER.info("Persisted %s cleaned records to SQLite", len(persist))


def save_model_metrics(metrics: pd.DataFrame, path: Path = DATABASE_PATH) -> None:
    with managed_connection(path) as db:
        metrics.to_sql("model_metrics", db, if_exists="replace", index=False)


def save_prediction(record: dict, path: Path = DATABASE_PATH) -> None:
    with managed_connection(path) as db:
        pd.DataFrame([record]).to_sql("prediction_history", db, if_exists="append", index=False)


def load_prediction_history(path: Path = DATABASE_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        with managed_connection(path) as db:
            return pd.read_sql_query("SELECT * FROM prediction_history ORDER BY created_at DESC", db)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
