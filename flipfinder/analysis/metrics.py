"""Agreement metrics between the algorithm and Sahil's eye."""
import numpy as np
from scipy.stats import spearmanr

from ..db import current_weights


def _rate(pairs):
    return round(sum(1 for ai, human in pairs if ai == human) / len(pairs), 3) if pairs else None


def ai_agreement(conn):
    """How often the human verdict matches the AI suggestion. The human label is
    the truth side of every comparison; shown vs blind is split out because a
    high agreement rate on shown labels alone may just be anchoring."""
    rows = conn.execute(
        """SELECT l.verdict, l.ai_shown, l.condition_human,
                  a.suggested_verdict, a.condition
           FROM labels l JOIN ai_labels a ON a.id = l.ai_label_id"""
    ).fetchall()
    pairs = [(r["suggested_verdict"], r["verdict"]) for r in rows]
    shown = [(r["suggested_verdict"], r["verdict"]) for r in rows if r["ai_shown"]]
    blind = [(r["suggested_verdict"], r["verdict"]) for r in rows if not r["ai_shown"]]
    confusion = [[0, 0, 0] for _ in range(3)]  # [ai_suggested][human_verdict]
    for ai, human in pairs:
        confusion[ai][human] += 1
    diffs = [
        abs(r["condition_human"] - r["condition"])
        for r in rows
        if r["condition_human"] is not None and r["condition_human"] != r["condition"]
    ]
    return {
        "n": len(rows),
        "verdict_agreement": _rate(pairs),
        "shown": {"n": len(shown), "verdict_agreement": _rate(shown)},
        "blind": {"n": len(blind), "verdict_agreement": _rate(blind)},
        "confusion": confusion,
        "condition_corrections": len(diffs),
        "condition_mae": round(float(np.mean(diffs)), 2) if diffs else None,
    }


def summary(conn, cfg):
    version, weights = current_weights(conn, cfg)
    counts = {
        r["verdict"]: r["c"]
        for r in conn.execute("SELECT verdict, COUNT(*) c FROM labels GROUP BY verdict")
    }
    total_scored = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]

    hold = conn.execute(
        """SELECT l.verdict, s.score, li.address, li.id listing_id
           FROM labels l
           JOIN scores s ON s.listing_id = l.listing_id
           JOIN listings li ON li.id = l.listing_id
           WHERE l.split='holdout'"""
    ).fetchall()

    spearman = p20 = None
    if len(hold) >= 3:
        scores = [r["score"] for r in hold]
        verdicts = [r["verdict"] for r in hold]
        if len(set(verdicts)) >= 2:
            rho, _ = spearmanr(scores, verdicts)
            spearman = None if np.isnan(rho) else round(float(rho), 3)
        order = np.argsort(scores)[::-1]
        k = min(20, len(order))
        p20 = round(sum(1 for i in order[:k] if verdicts[i] >= 1) / k, 3)

    labeled = conn.execute(
        """SELECT l.verdict, s.score, li.address, li.price, li.id listing_id
           FROM labels l
           JOIN scores s ON s.listing_id = l.listing_id
           JOIN listings li ON li.id = l.listing_id"""
    ).fetchall()
    algo_likes_you_pass = [
        {"listing_id": r["listing_id"], "address": r["address"],
         "price": r["price"], "score": round(r["score"], 2)}
        for r in labeled if r["verdict"] == 0 and r["score"] >= 0.75
    ][:10]
    you_like_algo_cold = [
        {"listing_id": r["listing_id"], "address": r["address"],
         "price": r["price"], "score": round(r["score"], 2)}
        for r in labeled if r["verdict"] == 2 and r["score"] <= 0.4
    ][:10]

    versions = [
        dict(r)
        for r in conn.execute(
            "SELECT version, train_n, holdout_spearman, holdout_p20, promoted, note, ts "
            "FROM scorer_versions ORDER BY version DESC LIMIT 5"
        )
    ]

    return {
        "scorer_version": version,
        "weights": {k: round(v, 3) for k, v in weights.items()},
        "labels": {
            "pass": counts.get(0, 0),
            "maybe": counts.get(1, 0),
            "deal": counts.get(2, 0),
            "total": sum(counts.values()),
        },
        "scored_listings": total_scored,
        "holdout_spearman": spearman,
        "holdout_p20": p20,
        "disagreements": {
            "algo_likes_you_pass": algo_likes_you_pass,
            "you_like_algo_cold": you_like_algo_cold,
        },
        "recent_versions": versions,
        "ai_agreement": ai_agreement(conn),
    }
