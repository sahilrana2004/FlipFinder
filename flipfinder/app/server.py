import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..analysis import fitter, metrics, photos
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
    ai_label_id: int | None = None  # the AI suggestion rendered with this listing, shown or hidden
    condition: int | None = None    # sent only when the user changed it


def _ai_label(conn, cfg, listing_id):
    row = conn.execute(
        """SELECT a.*, s.margin FROM ai_labels a
           LEFT JOIN scores s ON s.listing_id = a.listing_id
           WHERE a.listing_id=? AND a.model=? AND a.prompt_version=?
           ORDER BY a.id DESC LIMIT 1""",
        (listing_id, cfg["ollama"]["model"], photos.PROMPT_VERSION),
    ).fetchone()
    if not row:
        return None
    margin = row["margin"]
    return {
        "id": row["id"],
        "model": row["model"],
        "prompt_version": row["prompt_version"],
        "condition": row["condition"],
        "reno_scope": row["reno_scope"],
        "red_flags": json.loads(row["red_flags"]),
        "photos_representative": bool(row["photos_representative"]),
        "text_conflict": bool(row["text_conflict"]),
        "downgrade": bool(row["downgrade"]),
        # Derived from the current margin, never the stored column, so it can't be stale.
        "margin_ceiling": None if margin is None else photos.margin_ceiling(margin),
        "suggested_verdict": photos.ai_verdict(margin, row["downgrade"]),
        "reasons": json.loads(row["reasons"]),
        "notes": row["notes"],
    }


def _listing_payload(conn, cfg, row):
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
        "SELECT verdict, tags, condition_human FROM labels WHERE listing_id=?", (row["id"],)
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
        "condition_source": row["condition_source"],
        "distress_score": row["distress_score"],
        "distress_signals": json.loads(row["distress_signals"] or "{}"),
        "score": score["score"] if score else None,
        "components": json.loads(score["components"]) if score else None,
        "photos": photos,
        "label": dict(label) if label else None,
        "ai_label": _ai_label(conn, cfg, row["id"]),
    }




@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/listings")
def listings():
    cfg = load_config()
    conn = connect()
    rows = conn.execute(
        """SELECT l.id, l.address, l.price, l.beds, l.baths, l.sqft, l.year_built,
                  l.dom, l.lat, l.lng, s.score, s.margin,
                  l.condition_score AS condition, l.reno_scope,
                  a.downgrade, json_array_length(a.red_flags) AS red_flag_count,
                  a.text_conflict,
                  (SELECT COUNT(*) FROM labels WHERE listing_id=l.id) labeled
           FROM listings l JOIN scores s ON s.listing_id=l.id
           LEFT JOIN ai_labels a ON a.listing_id=l.id AND a.model=? AND a.prompt_version=?
           WHERE l.active=1 AND l.lat IS NOT NULL
           ORDER BY s.score DESC""",
        (cfg["ollama"]["model"], photos.PROMPT_VERSION),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["ai_verdict"] = photos.ai_verdict(d.pop("margin"), d.pop("downgrade"))
        if d["text_conflict"] is not None:
            d["text_conflict"] = bool(d["text_conflict"])
        out.append(d)
    conn.close()
    return out


@app.get("/api/listing/{listing_id}")
def listing(listing_id: int):
    cfg = load_config()
    conn = connect()
    row = conn.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "listing not found")
    out = _listing_payload(conn, cfg, row)
    conn.close()
    return out


@app.get("/api/triage/next")
def triage_next():
    """AI-labeled listings first, and among those the ones the model flagged
    (text conflict or red flags), since those most need a human eye. Within that
    tier: mostly boundary-uncertain listings, with 25% random exploration so the
    label set doesn't collapse around the threshold."""
    cfg = load_config()
    conn = connect()
    threshold = cfg["scoring"]["deal_threshold"]
    rows = conn.execute(
        """SELECT l.*, s.score,
                  CASE WHEN a.id IS NULL THEN 2
                       WHEN a.text_conflict=1 OR a.red_flags <> '[]' THEN 0
                       ELSE 1 END AS ai_priority
           FROM listings l
           JOIN scores s ON s.listing_id=l.id
           LEFT JOIN ai_labels a ON a.listing_id=l.id AND a.model=? AND a.prompt_version=?
           WHERE l.active=1
             AND l.id NOT IN (SELECT listing_id FROM labels)""",
        (cfg["ollama"]["model"], photos.PROMPT_VERSION),
    ).fetchall()
    if not rows:
        conn.close()
        return {"done": True}
    top = min(r["ai_priority"] for r in rows)
    pool = [r for r in rows if r["ai_priority"] == top]
    if random.random() < 0.25:
        pick = random.choice(pool)
    else:
        pick = min(pool, key=lambda r: abs(r["score"] - threshold))
    out = _listing_payload(conn, cfg, pick)
    remaining = len(rows)
    conn.close()
    out["remaining"] = remaining
    return out


@app.post("/api/labels")
def add_label(body: LabelIn):
    if body.verdict not in (0, 1, 2):
        raise HTTPException(400, "verdict must be 0, 1, or 2")
    if body.condition is not None and not 1 <= body.condition <= 10:
        raise HTTPException(400, "condition must be 1-10")
    cfg = load_config()
    conn = connect()
    score = conn.execute(
        "SELECT score, components FROM scores WHERE listing_id=?", (body.listing_id,)
    ).fetchone()
    if not score:
        conn.close()
        raise HTTPException(400, "listing has no score — run scoring first")
    if body.ai_label_id is not None and not conn.execute(
        "SELECT 1 FROM ai_labels WHERE id=? AND listing_id=?",
        (body.ai_label_id, body.listing_id),
    ).fetchone():
        conn.close()
        raise HTTPException(400, "ai_label_id does not belong to this listing")
    ai_shown = int(body.ai_label_id is not None and not body.blind)
    version, _ = current_weights(conn, cfg)
    split = (
        "holdout"
        if random.random() < cfg["fitting"]["holdout_fraction"]
        else "train"
    )
    # ai_shown only ratchets up on relabel: once the suggestion was seen, the
    # verdict can't be treated as blind anymore.
    conn.execute(
        """INSERT INTO labels
             (listing_id, verdict, tags, feature_snapshot, algo_score_shown,
              scorer_version, split, ai_label_id, ai_shown, condition_human)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(listing_id) DO UPDATE SET
             verdict=excluded.verdict, tags=excluded.tags,
             ai_label_id=COALESCE(excluded.ai_label_id, labels.ai_label_id),
             ai_shown=MAX(labels.ai_shown, excluded.ai_shown),
             condition_human=COALESCE(excluded.condition_human, labels.condition_human),
             ts=CURRENT_TIMESTAMP""",
        (body.listing_id, body.verdict, json.dumps(body.tags), score["components"],
         None if body.blind else score["score"], version, split,
         body.ai_label_id, ai_shown, body.condition),
    )
    if body.condition is not None:
        conn.execute(
            "UPDATE listings SET condition_score=?, condition_source='human' WHERE id=?",
            (body.condition, body.listing_id),
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
