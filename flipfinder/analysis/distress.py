"""Distress score per listing: keywords + price/sqft z-score + age + photo condition."""
import json


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def compute(conn, cfg):
    d = cfg["distress"]
    keywords = [k.lower() for k in d["keywords"]]
    thresh = d["ppsf_zscore_threshold"]
    old_year = d["old_year"]

    rows = conn.execute("SELECT * FROM listings WHERE active=1").fetchall()
    n = 0
    for row in rows:
        signals = {}

        text = (row["remarks"] or "").lower()
        hits = [k for k in keywords if k in text]
        kw = _clamp(len(hits) * 0.5)
        signals["keywords"] = hits

        # area_stats' $/sqft is close dollars (see census.refresh_market_stats); the
        # listing side is necessarily its asking $/sqft, so a negative z means asked
        # below what the tract actually closes for — which is the signal wanted.
        z_sig = 0.0
        stats = None
        if row["tract"]:
            stats = conn.execute(
                "SELECT median_sold_ppsf, std_sold_ppsf FROM area_stats WHERE tract=?",
                (row["tract"],),
            ).fetchone()
        if stats and stats["median_sold_ppsf"] and stats["std_sold_ppsf"] and row["ppsf"]:
            z = (row["ppsf"] - stats["median_sold_ppsf"]) / stats["std_sold_ppsf"]
            z_sig = _clamp((thresh - z) / 1.5)
            signals["ppsf_z"] = round(z, 2)

        age_sig = 1.0 if (row["year_built"] and row["year_built"] < old_year) else 0.0
        signals["old"] = bool(age_sig)

        photo_sig = 0.0
        if row["condition_score"] is not None:
            photo_sig = _clamp((6.0 - row["condition_score"]) / 5.0)
            signals["photo_condition"] = row["condition_score"]

        score = _clamp(0.45 * kw + 0.30 * z_sig + 0.10 * age_sig + 0.35 * photo_sig)
        conn.execute(
            "UPDATE listings SET distress_score=?, distress_signals=? WHERE id=?",
            (score, json.dumps(signals), row["id"]),
        )
        n += 1
    conn.commit()
    return n
