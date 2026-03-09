"""
Kila Int -- FastAPI Backend
============================
Serves intelligence data from SQLite to the tactical map frontend.

Run:  python server.py
Then: http://localhost:8000
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import parser as intel_parser

app = FastAPI(title="Kila Int API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Faction heuristic -- derive from entities/keywords in parsed intel
# ---------------------------------------------------------------------------
IRAN_MARKERS = {"iran", "irgc", "hezbollah", "houthis", "ansar allah", "quds force",
                "hamas", "pmc wagner", "wagner"}
ISRAEL_MARKERS = {"israel", "idf", "iof", "mossad", "centcom", "us", "usa",
                  "american", "coalition"}


def guess_faction(entities: list[dict], text: str) -> str:
    """Guess iran/israel/unknown from entities and text content."""
    entity_names = {e["name"].lower() for e in entities}
    text_lower = (text or "").lower()

    iran_score = sum(1 for m in IRAN_MARKERS if m in entity_names or m in text_lower)
    israel_score = sum(1 for m in ISRAEL_MARKERS if m in entity_names or m in text_lower)

    if iran_score > israel_score:
        return "iran"
    elif israel_score > iran_score:
        return "israel"
    return "unknown"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/events")
def get_events(
    hours: int = Query(72, ge=1, le=8760),
    event_type: str | None = Query(None),
    channel: str | None = Query(None),
):
    """Return geo-tagged intelligence events for the map."""
    conn = db.get_connection()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        query = """
            SELECT m.id, m.channel_id, m.channel_name, m.message_id, m.sender_name,
                   m.text, m.date, m.media_type,
                   p.event_type, p.sentiment, p.locations, p.entities, p.keywords,
                   COALESCE(p.relevance, 0.0) as relevance
            FROM messages m
            LEFT JOIN parsed_intel p ON p.message_rowid = m.id
            WHERE m.date >= ?
        """
        params: list = [cutoff]

        if event_type:
            query += " AND p.event_type = ?"
            params.append(event_type)
        if channel:
            query += " AND m.channel_name LIKE ?"
            params.append(f"%{channel}%")

        query += " ORDER BY m.date DESC"
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    events = []
    for row in rows:
        row = dict(row)
        locations = json.loads(row["locations"]) if row["locations"] else []
        entities = json.loads(row["entities"]) if row["entities"] else []
        keywords = json.loads(row["keywords"]) if row["keywords"] else []

        # Skip low-relevance messages (noise/rhetoric without concrete intel)
        if row["relevance"] < 0.20:
            continue

        geo = intel_parser.geocode_locations(locations)
        if not geo:
            continue  # skip messages without mappable locations

        faction = guess_faction(entities, row["text"])
        event_type = row["event_type"]

        # Use the first geocoded location as the marker position
        primary = geo[0]

        events.append({
            "id": row["id"],
            "lat": primary["lat"],
            "lng": primary["lng"],
            "loc": primary["name"],
            "desc": (row["text"] or "")[:300],
            "f": faction,
            "event_type": event_type,
            "sentiment": row["sentiment"],
            "keywords": keywords,
            "entities": entities,
            "all_locations": [g["name"] for g in geo],
            "src": [{
                "ch": row["channel_name"] or "unknown",
                "msg": str(row["message_id"]),
                "t": int(datetime.fromisoformat(row["date"]).timestamp() * 1000),
            }],
            "t": int(datetime.fromisoformat(row["date"]).timestamp() * 1000),
            "channel_name": row["channel_name"],
        })

    return events


@app.get("/api/channels")
def get_channels():
    """Return list of channels with message counts."""
    conn = db.get_connection()
    try:
        rows = conn.execute("""
            SELECT channel_name, channel_id, COUNT(*) as count,
                   MAX(date) as last_message
            FROM messages
            GROUP BY channel_id
            ORDER BY count DESC
        """).fetchall()
    finally:
        conn.close()

    colors = ["#e04040", "#4488ee", "#a0c830", "#9070d0", "#e07020", "#20b0a0"]
    return [
        {
            "id": row["channel_name"] or str(row["channel_id"]),
            "name": row["channel_name"] or str(row["channel_id"]),
            "color": colors[i % len(colors)],
            "count": row["count"],
            "last_message": row["last_message"],
            "on": True,
        }
        for i, row in enumerate(rows)
    ]


@app.get("/api/stats")
def get_stats(hours: int = Query(72, ge=1, le=8760)):
    """Return summary statistics."""
    conn = db.get_connection()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE date >= ?", (cutoff,)
        ).fetchone()[0]

        geo_count = 0
        event_types = {}
        rows = conn.execute("""
            SELECT p.event_type, p.locations, p.entities, m.text
            FROM messages m
            LEFT JOIN parsed_intel p ON p.message_rowid = m.id
            WHERE m.date >= ?
        """, (cutoff,)).fetchall()

        iran_count = 0
        israel_count = 0
        for row in rows:
            row = dict(row)
            locations = json.loads(row["locations"]) if row["locations"] else []
            if locations:
                geo_count += 1
            entities = json.loads(row["entities"]) if row["entities"] else []
            faction = guess_faction(entities, row["text"])
            if faction == "iran":
                iran_count += 1
            elif faction == "israel":
                israel_count += 1

            et = row["event_type"]
            if et:
                event_types[et] = event_types.get(et, 0) + 1
    finally:
        conn.close()

    return {
        "total_messages": total,
        "geo_tagged": geo_count,
        "iran": iran_count,
        "israel": israel_count,
        "unknown": total - iran_count - israel_count,
        "event_types": event_types,
    }


@app.get("/")
def serve_map():
    """Serve the tactical map."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


if __name__ == "__main__":
    db.init_db()
    port = int(os.environ.get("PORT", 8000))
    print(f"[Kila Int] DB: {db.DB_PATH} ({db.get_message_count()} messages)")
    print(f"[Kila Int] Starting server at http://localhost:{port}")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
