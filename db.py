"""
Kila Int -- SQLite storage layer
=================================
Stores ingested Telegram messages and parsed intelligence data.
Deduplicates on (channel_id, message_id) so re-runs are safe.
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kila.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id      INTEGER NOT NULL,
    channel_name    TEXT,
    message_id      INTEGER NOT NULL,
    sender_name     TEXT,
    text            TEXT,
    date            TEXT NOT NULL,
    media_type      TEXT,
    ingested_at     TEXT NOT NULL,
    UNIQUE(channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS parsed_intel (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_rowid   INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    event_type      TEXT,
    sentiment       REAL,
    locations       TEXT,   -- JSON array of strings
    entities        TEXT,   -- JSON array of {name, label} objects
    keywords        TEXT,   -- JSON array of strings
    relevance       REAL DEFAULT 0.0,
    parsed_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);
CREATE INDEX IF NOT EXISTS idx_parsed_event ON parsed_intel(event_type);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.close()


def insert_message(
    channel_id: int,
    channel_name: str,
    message_id: int,
    sender_name: str | None,
    text: str | None,
    date: datetime,
    media_type: str | None,
) -> int | None:
    """Insert a message. Returns the row id, or None if it was a duplicate."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO messages
               (channel_id, channel_name, message_id, sender_name, text, date, media_type, ingested_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                channel_id,
                channel_name,
                message_id,
                sender_name,
                text,
                date.isoformat(),
                media_type,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            return None  # duplicate
        return cur.lastrowid
    finally:
        conn.close()


def insert_parsed_intel(
    message_rowid: int,
    event_type: str | None,
    sentiment: float | None,
    locations: list[str],
    entities: list[dict],
    keywords: list[str],
    relevance: float = 0.0,
):
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO parsed_intel
               (message_rowid, event_type, sentiment, locations, entities, keywords, relevance, parsed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message_rowid,
                event_type,
                sentiment,
                json.dumps(locations),
                json.dumps(entities),
                json.dumps(keywords),
                relevance,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_message_count() -> int:
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def get_recent_messages(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT m.*, p.event_type, p.sentiment, p.locations, p.entities, p.keywords
               FROM messages m
               LEFT JOIN parsed_intel p ON p.message_rowid = m.id
               ORDER BY m.date DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
