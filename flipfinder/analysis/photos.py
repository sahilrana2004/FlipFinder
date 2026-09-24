"""AI labeling via a local vision model (Ollama). The model reads each scored
listing's photos, remarks, and numbers and returns condition, reno scope, red
flags, and whether to downgrade. The verdict itself is derived from the current
margin (see ai_verdict). Output goes to `ai_labels`, never `labels`: AI verdicts
must not become training data for the fitter.
"""
import base64
import json
import re
import time

import requests

from ..config import ROOT
from ..ingest.redfin import UA

PROMPT_VERSION = "v8"
VERDICTS = {"pass": 0, "maybe": 1, "deal": 2}
VERDICT_NAMES = ("pass", "maybe", "deal")
SCOPES = ("light", "medium", "gut")
FAILURE_LOG = ROOT / "data" / "autolabel" / "failures.log"

# v3 asked for an absolute verdict: 28 of 33 answers just restated the margin
# ceiling, and verdicts went stale once the model's condition changed the reno
# tier. v4-v7 asked the model for a downgrade decision instead, and across four
# prompt revisions the same pilot listings flipped between versions. So the model
# now only reports what it sees; the downgrade is derived here from its red flags
# and text_conflict, and the verdict from whatever the margin is at read time.

# Risks that reno_scope's cost per sqft doesn't price in. An as-is sale is
# deliberately absent: on its own it's a sale term, not a defect.
MATERIAL_FLAGS = (
    ("foundation", re.compile(r"foundation|structural", re.I)),
    ("water damage", re.compile(r"water (damage|intrusion|stain)|leak|flood", re.I)),
    ("fire", re.compile(r"\bfire\b|smoke damage", re.I)),
    ("mold", re.compile(r"\bmold|mildew", re.I)),
    ("roof", re.compile(r"\broof", re.I)),
    ("unfinished work", re.compile(r"unfinished|incomplete|partially (completed|finished)", re.I)),
    ("tenant occupied", re.compile(r"tenant", re.I)),
)

PROMPT = """You are screening a single-family house for a fix-and-flip investor. \
Be specific and do not guess optimistically.

LISTING
Ask price: {price}
Size: {sqft} sqft, {beds} beds / {baths} baths, built {year_built}
Days on market: {dom}

APP ESTIMATES (from sold comps)
ARV: {arv}
Current margin after reno, carry and closing: {margin}
Comps used: {comp_count}

LISTING REMARKS
\"\"\"{remarks}\"\"\"

The {n_photos} attached images are the first listing photos. Listing photos are chosen by the \
seller's agent to flatter the house: they favor the best rooms, stage and brighten them, and \
leave out damage. A room that is not shown is not evidence that it is fine.

Reply with JSON containing exactly these keys, in this order:
- "notes": two sentences. First: what the remarks literally say about the house's physical \
condition, quoting the words, or "Remarks do not describe condition." Second: what a buyer \
should check.
- "condition": integer 1-10 rating ONLY what the photos show. 1-3 gutted, major damage, or \
unlivable; 4-6 dated or worn but livable; 7-8 updated with minor work left; 9-10 fully \
renovated and move-in ready.
- "reno_scope": "light", "medium", or "gut" - the work the house actually needs, judged from \
photos and remarks together. Light is paint, flooring, fixtures. Medium adds kitchen or baths. \
Gut is down to the studs, or major systems and structure.
- "red_flags": list of short strings, only for material issues that change cost, risk, or \
resale value, seen in photos or stated in remarks: foundation, water damage, fire, roof, mold, \
unfinished renovation, tenant occupied, as-is sale, and similar. Only list what you actually \
see or what the remarks actually say - never something unknown, unshown, or possible ("roof \
age unknown", "bathrooms not pictured"). Never cosmetic taste (paint color, carpet, staging, \
landscaping, dated finishes). Empty list if none.
- "photos_representative": true if the photos plausibly show the whole house's real condition; \
false if they are few, exterior-only, or skip rooms that matter.
- "text_conflict": true ONLY if the words you quoted in notes describe a materially different \
physical condition than the photos show - for example photos look finished but remarks say \
"needs full rehab", "down to the studs", or "renovation not complete"; or photos look rough but \
remarks say "fully remodeled". False when remarks do not describe condition. Marketing words \
("potential", "opportunity", "make it your own", "investor", "as-is") are not a condition \
description and never make this true. Dated finishes, older fixtures, or cosmetic wear that \
the remarks don't contradict are not a conflict either."""


def _money(v):
    return "unknown" if v is None else f"${v:,.0f}"


def _num(v):
    return "unknown" if v is None else f"{v:g}"


def margin_ceiling(margin):
    """Highest verdict the app's margin supports: negative is pass, up to 10% maybe."""
    if margin < 0:
        return VERDICTS["pass"]
    return VERDICTS["maybe"] if margin <= 0.10 else VERDICTS["deal"]


def downgrade_reasons(red_flags, text_conflict):
    """Why the verdict drops one step below the margin ceiling; empty means no
    downgrade. Each reason carries the model's own flag text so it can be checked."""
    reasons = [
        f"{name}: {flag}"
        for flag in red_flags
        for name, pattern in MATERIAL_FLAGS
        if pattern.search(flag)
    ]
    if text_conflict:
        reasons.append("remarks describe a different condition than the photos show")
    return list(dict.fromkeys(reasons))[:3]


def ai_verdict(margin, downgrade):
    """The AI verdict, always derived from the current margin so it can't go stale
    when a re-score moves the margin. None when unscored or not labeled with a
    downgrade decision."""
    if margin is None or downgrade is None:
        return None
    return max(0, margin_ceiling(margin) - downgrade)


def refresh_verdicts(conn, cfg):
    """Rewrite stored suggested_verdict for current-version labels from current
    margins, so the Metrics agreement query (which reads the column) stays true."""
    rows = conn.execute(
        """SELECT a.id, a.downgrade, s.margin FROM ai_labels a
           JOIN scores s ON s.listing_id = a.listing_id
           WHERE a.model = ? AND a.prompt_version = ?""",
        (cfg["ollama"]["model"], PROMPT_VERSION),
    ).fetchall()
    for r in rows:
        conn.execute("UPDATE ai_labels SET suggested_verdict=? WHERE id=?",
                     (ai_verdict(r["margin"], r["downgrade"]), r["id"]))
    conn.commit()
    return len(rows)


def build_prompt(listing, components, n_photos):
    c = components or {}
    margin = c.get("margin")
    return PROMPT.format(
        price=_money(listing["price"]),
        sqft=_num(listing["sqft"]),
        beds=_num(listing["beds"]),
        baths=_num(listing["baths"]),
        year_built=_num(listing["year_built"]),
        dom=_num(listing["dom"]),
        arv=_money(c.get("arv")),
        margin="unknown" if margin is None else f"{margin * 100:.1f}%",
        comp_count=c.get("comp_count", "unknown"),
        remarks=(listing["remarks"] or "(none)")[:1500],
        n_photos=n_photos,
    )


def ollama_available(cfg):
    try:
        r = requests.get(f'{cfg["ollama"]["url"]}/api/tags', timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _fetch_images(urls):
    images = []
    for url in urls:
        try:
            resp = requests.get(url, headers=UA, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            images.append(base64.b64encode(resp.content).decode())
    return images


def _generate(cfg, prompt, images):
    # qwen3.5 is a thinking model: without think=false responses are slow and
    # wrapped in reasoning text that breaks format=json. Some listings carry
    # full-size "bigphoto" images; four of those plus the prompt overflow
    # Ollama's default 4096-token context and fail with HTTP 400.
    r = requests.post(
        f'{cfg["ollama"]["url"]}/api/generate',
        json={
            "model": cfg["ollama"]["model"],
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"num_ctx": 16384},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def _flag(v):
    if isinstance(v, bool):
        return int(v)
    if v in (0, 1):
        return int(v)
    raise ValueError(f"expected boolean, got {v!r}")


def _str_list(v, name):
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{name} must be a list of strings")
    return [x.strip() for x in v if x.strip()]


def parse_response(text):
    """Returns validated fields, or raises ValueError on anything off-spec."""
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("response is not a JSON object")

    cond = out.get("condition")
    if isinstance(cond, bool) or not isinstance(cond, (int, float)) or cond != int(cond):
        raise ValueError(f"condition must be an integer, got {cond!r}")
    cond = int(cond)
    if not 1 <= cond <= 10:
        raise ValueError(f"condition out of range: {cond}")

    scope = out.get("reno_scope")
    if scope not in SCOPES:
        raise ValueError(f"reno_scope invalid: {scope!r}")
    notes = out.get("notes")
    # The prompt asks for two sentences in notes; the model sometimes returns them as a list.
    if isinstance(notes, list):
        notes = " ".join(_str_list(notes, "notes"))
    if notes is not None and not isinstance(notes, str):
        raise ValueError("notes must be a string")

    red_flags = _str_list(out.get("red_flags"), "red_flags")
    text_conflict = _flag(out.get("text_conflict"))
    reasons = downgrade_reasons(red_flags, text_conflict)
    return {
        "condition": cond,
        "reno_scope": scope,
        "red_flags": red_flags,
        "photos_representative": _flag(out.get("photos_representative")),
        "text_conflict": text_conflict,
        "downgrade": int(bool(reasons)),
        "reasons": reasons,
        "notes": (notes or "").strip(),
    }


def _log_failure(listing, error, raw):
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG, "a", encoding="utf-8") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} listing {listing['id']} "
                f"{listing['address']}\nerror: {error}\nraw: {raw}\n")


def eligible(conn, cfg, limit=None):
    """Unlabeled scored listings with photos, best score first. At 15-60s a listing
    a full buy box is hours of GPU time, so callers cap the run with `limit`; the
    next run picks up where this one stopped, since labeled rows drop out here."""
    sql = """SELECT l.*, s.components FROM listings l
             JOIN scores s ON s.listing_id = l.id
             WHERE EXISTS (SELECT 1 FROM photos p WHERE p.listing_id = l.id)
               AND NOT EXISTS (SELECT 1 FROM ai_labels a WHERE a.listing_id = l.id
                               AND a.model = ? AND a.prompt_version = ?)
             ORDER BY s.score DESC"""
    params = [cfg["ollama"]["model"], PROMPT_VERSION]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def label_listing(conn, cfg, listing):
    """Label one listing. Returns the stored fields plus seconds, or None after
    logging a failure (retried once on bad output)."""
    start = time.monotonic()
    urls = [
        r["url"] for r in conn.execute(
            "SELECT url FROM photos WHERE listing_id=? ORDER BY rowid LIMIT ?",
            (listing["id"], cfg["ollama"]["photos_per_listing"]),
        )
    ]
    images = _fetch_images(urls)
    if not images:
        _log_failure(listing, "no photo could be downloaded", "")
        return None
    components = json.loads(listing["components"] or "{}")
    prompt = build_prompt(listing, components, len(images))

    raw, fields, error = "", None, None
    for _ in range(2):
        try:
            raw = _generate(cfg, prompt, images)
            fields = parse_response(raw)
            break
        except (requests.RequestException, KeyError, ValueError) as e:
            error = e
    if fields is None:
        _log_failure(listing, error, raw)
        return None

    seconds = time.monotonic() - start
    # Provisional: the margin moves once reno_scope is applied, and
    # refresh_verdicts rewrites this after the re-score.
    verdict = ai_verdict(components.get("margin"), fields["downgrade"])
    cur = conn.execute(
        """INSERT INTO ai_labels
             (listing_id, model, prompt_version, condition, reno_scope, red_flags,
              photos_representative, text_conflict, downgrade, suggested_verdict, reasons,
              notes, photo_count, seconds, raw_response)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (listing["id"], cfg["ollama"]["model"], PROMPT_VERSION, fields["condition"],
         fields["reno_scope"], json.dumps(fields["red_flags"]),
         fields["photos_representative"], fields["text_conflict"], fields["downgrade"],
         verdict, json.dumps(fields["reasons"]), fields["notes"], len(images), seconds, raw),
    )
    # A human-verified condition always wins over the model's.
    conn.execute(
        """UPDATE listings SET condition_score=?, reno_scope=?, condition_source='ai'
           WHERE id=? AND condition_source IS NOT 'human'""",
        (fields["condition"], fields["reno_scope"], listing["id"]),
    )
    conn.commit()
    return {**fields, "id": cur.lastrowid, "seconds": seconds}


def label_all(conn, cfg, limit=None):
    """Yields (index, total, listing, result) per listing; result is None on
    failure. Resumable: already-labeled listings aren't eligible."""
    rows = eligible(conn, cfg, limit=limit)
    for i, listing in enumerate(rows, 1):
        yield i, len(rows), listing, label_listing(conn, cfg, listing)
