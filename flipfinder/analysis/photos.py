"""Photo condition scoring via a local vision model (Ollama). Optional stage:
if Ollama isn't running, the pipeline skips it and distress falls back to
keywords/price/age signals.
"""
import base64
import json

import requests

from ..ingest.redfin import UA

PROMPT = (
    "You are assessing real-estate listing photos to estimate renovation scope. "
    "Rate the overall interior/exterior condition from 1 (gutted, severe damage, "
    "uninhabitable) to 10 (fully renovated, move-in ready). Dated-but-clean is 5-6. "
    'Reply with JSON only: {"condition": <number>}'
)


def ollama_available(cfg):
    try:
        r = requests.get(f'{cfg["ollama"]["url"]}/api/tags', timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _score_images(cfg, images_b64):
    r = requests.post(
        f'{cfg["ollama"]["url"]}/api/generate',
        json={
            "model": cfg["ollama"]["model"],
            "prompt": PROMPT,
            "images": images_b64,
            "stream": False,
            "format": "json",
        },
        timeout=180,
    )
    r.raise_for_status()
    out = json.loads(r.json()["response"])
    cond = float(out["condition"])
    if not 1 <= cond <= 10:
        raise ValueError(f"condition out of range: {cond}")
    return cond


def score_all(conn, cfg, limit=60):
    if not ollama_available(cfg):
        return 0, "ollama not reachable — skipped photo scoring"
    per = cfg["ollama"]["photos_per_listing"]
    rows = conn.execute(
        """SELECT DISTINCT l.id FROM listings l JOIN photos p ON p.listing_id=l.id
           WHERE l.active=1 AND l.condition_score IS NULL LIMIT ?""",
        (limit,),
    ).fetchall()
    scored = 0
    for row in rows:
        urls = [
            r["url"]
            for r in conn.execute(
                "SELECT url FROM photos WHERE listing_id=? LIMIT ?", (row["id"], per)
            )
        ]
        images = []
        for url in urls:
            try:
                resp = requests.get(url, headers=UA, timeout=30)
                if resp.status_code == 200:
                    images.append(base64.b64encode(resp.content).decode())
            except requests.RequestException:
                continue
        if not images:
            continue
        try:
            cond = _score_images(cfg, images)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
            continue
        conn.execute(
            "UPDATE listings SET condition_score=? WHERE id=?", (cond, row["id"])
        )
        conn.commit()
        scored += 1
    return scored, "ok"
