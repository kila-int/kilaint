"""
Re-parse all existing messages with the updated parser.
Deletes old parsed_intel rows and re-inserts with new scoring.

Usage: python reparse.py
"""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import db
import parser as intel_parser

def main():
    db.init_db()

    # Add relevance column if it doesn't exist yet (existing DB migration)
    conn = db.get_connection()
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(parsed_intel)").fetchall()]
        if "relevance" not in cols:
            conn.execute("ALTER TABLE parsed_intel ADD COLUMN relevance REAL DEFAULT 0.0")
            conn.commit()
            print("[migrate] Added 'relevance' column to parsed_intel")
    finally:
        conn.close()

    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT id, text FROM messages").fetchall()
    finally:
        conn.close()

    print(f"[reparse] {len(rows)} messages to process")

    # Clear old parsed data
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM parsed_intel")
        conn.commit()
    finally:
        conn.close()

    classified = 0
    geo_tagged = 0
    relevant = 0

    for row in rows:
        row = dict(row)
        text = row["text"]
        if not text:
            continue

        parsed = intel_parser.parse_message(text)
        db.insert_parsed_intel(
            message_rowid=row["id"],
            event_type=parsed["event_type"],
            sentiment=parsed["sentiment"],
            locations=parsed["locations"],
            entities=parsed["entities"],
            keywords=parsed["keywords"],
            relevance=parsed.get("relevance", 0.0),
        )

        if parsed["event_type"]:
            classified += 1
        if parsed["locations"]:
            geo_tagged += 1
        if parsed.get("relevance", 0) >= 0.30:
            relevant += 1

    total = len(rows)
    print(f"\n[reparse] Done!")
    print(f"  Total:      {total}")
    print(f"  Classified: {classified} ({100*classified//max(total,1)}%)")
    print(f"  Geo-tagged: {geo_tagged} ({100*geo_tagged//max(total,1)}%)")
    print(f"  Relevant:   {relevant} ({100*relevant//max(total,1)}%)")


if __name__ == "__main__":
    main()
