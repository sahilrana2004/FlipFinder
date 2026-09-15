import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..analysis import fitter, metrics
from ..config import load_config
from ..db import connect, current_weights, init_db

app = FastAPI(title="FlipFinder")
STATIC = Path(__file__).parent / "static"

init_db()


class LabelIn(BaseModel):
    listing_id: int
    verdict: int  # 0=pass 1=maybe 2=deal
    tags: list[str] = []
    blind: bool = False


def _listing_payload(conn, row):
    score = conn.execute(
        "SELECT * FROM scores WHERE listing_id=?", (row["id"],)
    ).fetchone()
    photos = [
        r["url"]
        for r in conn.execute(
            "SELECT url FROM photos WHERE listing_id=? LIMIT 6", (row["id"],)
        )
    ]
    label = conn.execute(
        "SELECT verdict, tags FROM labels WHERE listing_id=?", (row["id"],)
    ).fetchone()
    return {
        "id": row["id"],
        "address": row["address"],
        "city": row["city"],
        "zip": row["zip"],
        "price": row["price"],
        "beds": row["beds"],
        "baths": row["baths"],
        "sqft": row["sqft"],
        "year_built": row["year_built"],
        "dom": row["dom"],
        "ppsf": row["ppsf"],
        "lat": row["lat"],
        "lng": row["lng"],
        "url": row["url"],
        "remarks": row["remarks"],
        "condition_score": row["condition_score"],
        "distress_score": row["distress_score"],
        "distress_signals": json.loads(row["distress_signals"] or "{}"),
        "score": score["score"] if score else None,
        "components": json.loads(score["components"]) if score else None,
        "photos": photos,
        "label": dict(label) if label else None,
    }




@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/listings")
def listings():
    conn = connect()
    rows = conn.execute(
        """SELECT l.id, l.address, l.price, l.beds, l.baths, l.sqft, l.year_built,
                  l.dom, l.lat, l.lng, s.score,
                  (SELECT COUNT(*) FROM labels WHERE listing_id=l.id) labeled
           FROM listings l JOIN scores s ON s.listing_id=l.id
           WHERE l.active=1 AND l.lat IS NOT NULL
           ORDER BY s.score DESC"""
    ).fetchall()
    out = [dict(r) for r in rows]
    conn.close()
    return out


@app.get("/api/listing/{listing_id}")
def listing(listing_id: int):
    conn = connect()
    row = conn.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "listing not found")
    out = _listing_payload(conn, row)
    conn.close()
    return out


@app.get("/api/triage/next")
def triage_next():
    """Active-learning order: mostly boundary-uncertain listings, with 25%
    random exploration so the label set doesn't collapse around the threshold."""
    cfg = load_config()
    conn = connect()
    threshold = cfg["scoring"]["deal_threshold"]
    rows = conn.execute(
        """SELECT l.*, s.score FROM listings l
           JOIN scores s ON s.listing_id=l.id
           WHERE l.active=1
             AND l.id NOT IN (SELECT listing_id FROM labels)"""
    ).fetchall()
    if not rows:
        conn.close()
        return {"done": True}
    if random.random() < 0.25:
        pick = random.choice(rows)
    else:
        pick = min(rows, key=lambda r: abs(r["score"] - threshold))
    out = _listing_payload(conn, pick)
    remaining = len(rows)
    conn.close()
    out["remaining"] = remaining
    return out


@app.post("/api/labels")
def add_label(body: LabelIn):
    if body.verdict not in (0, 1, 2):
        raise HTTPException(400, "verdict must be 0, 1, or 2")
    cfg = load_config()
    conn = connect()
    score = conn.execute(
        "SELECT score, components FROM scores WHERE listing_id=?", (body.listing_id,)
    ).fetchone()
    if not score:
        conn.close()
        raise HTTPException(400, "listing has no score — run scoring first")
    version, _ = current_weights(conn, cfg)
    split = (
        "holdout"
        if random.random() < cfg["fitting"]["holdout_fraction"]
        else "train"
    )
    conn.execute(
        """INSERT INTO labels
             (listing_id, verdict, tags, feature_snapshot, algo_score_shown,
              scorer_version, split)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(listing_id) DO UPDATE SET
             verdict=excluded.verdict, tags=excluded.tags,
             ts=CURRENT_TIMESTAMP""",
        (body.listing_id, body.verdict, json.dumps(body.tags), score["components"],
         None if body.blind else score["score"], version, split),
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) c FROM labels").fetchone()["c"]
    conn.close()
    return {"ok": True, "total_labels": total}


@app.get("/api/metrics")
def get_metrics():
    cfg = load_config()
    conn = connect()
    out = metrics.summary(conn, cfg)
    conn.close()
    return out


@app.post("/api/fit")
def fit():
    cfg = load_config()
    conn = connect()
    out = fitter.run(conn, cfg)
    conn.close()
    return out
