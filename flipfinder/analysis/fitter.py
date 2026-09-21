"""Fit scorer weights to Sahil's labels.

Tier 1: Nelder-Mead over the four weight exponents, maximizing Spearman rank
agreement between the score and verdicts on the train split. A new version is
promoted only if holdout agreement doesn't regress past the configured
tolerance — same gate pattern as the Carely eval pipeline.
"""
import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from ..db import current_weights
from .scorer import FEATURE_KEYS, combine, score_all


def _labeled_features(conn, split):
    rows = conn.execute(
        "SELECT verdict, feature_snapshot FROM labels WHERE split=? "
        "AND feature_snapshot IS NOT NULL",
        (split,),
    ).fetchall()
    feats, verdicts = [], []
    for r in rows:
        snap = json.loads(r["feature_snapshot"])
        if all(k in snap for k in FEATURE_KEYS) and "size_mult" in snap:
            feats.append(snap)
            verdicts.append(r["verdict"])
    return feats, verdicts


def _scores_for(weights, feats, cfg):
    return [combine(f, weights, cfg) for f in feats]


def _spearman(weights, feats, verdicts, cfg):
    if len(set(verdicts)) < 2:
        return 0.0
    rho, _ = spearmanr(_scores_for(weights, feats, cfg), verdicts)
    return 0.0 if np.isnan(rho) else float(rho)


def _precision_at_20(weights, feats, verdicts, cfg):
    if not feats:
        return None
    order = np.argsort(_scores_for(weights, feats, cfg))[::-1]
    k = min(20, len(order))
    top = [verdicts[i] for i in order[:k]]
    return sum(1 for v in top if v >= 1) / k


def _to_weights(x):
    return {k: float(np.clip(v, 0.0, 3.0)) for k, v in zip(FEATURE_KEYS, x)}


def run(conn, cfg):
    fit_cfg = cfg["fitting"]
    train_f, train_v = _labeled_features(conn, "train")
    hold_f, hold_v = _labeled_features(conn, "holdout")

    if len(train_f) < fit_cfg["min_train_labels"]:
        return {
            "promoted": False,
            "note": f"need >= {fit_cfg['min_train_labels']} train labels, "
                    f"have {len(train_f)}",
        }

    prev_version, prev_weights = current_weights(conn, cfg)
    x0 = np.array([prev_weights[k] for k in FEATURE_KEYS])

    res = minimize(
        lambda x: -_spearman(_to_weights(x), train_f, train_v, cfg),
        x0,
        method="Nelder-Mead",
        options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-4},
    )
    new_weights = _to_weights(res.x)

    train_sp = _spearman(new_weights, train_f, train_v, cfg)
    enough_holdout = len(hold_f) >= fit_cfg["min_holdout_labels"]
    hold_sp_new = _spearman(new_weights, hold_f, hold_v, cfg) if enough_holdout else None
    hold_sp_prev = _spearman(prev_weights, hold_f, hold_v, cfg) if enough_holdout else None
    hold_p20 = _precision_at_20(new_weights, hold_f, hold_v, cfg) if enough_holdout else None

    if not enough_holdout:
        promoted = False
        note = f"holdout too small ({len(hold_f)}) — label more before promoting"
    elif hold_sp_new >= hold_sp_prev - fit_cfg["max_regression"]:
        promoted = True
        note = (f"promoted: holdout spearman {hold_sp_prev:.3f} -> {hold_sp_new:.3f}")
    else:
        promoted = False
        note = (f"rejected: holdout spearman regressed "
                f"{hold_sp_prev:.3f} -> {hold_sp_new:.3f}")

    cur = conn.execute(
        """INSERT INTO scorer_versions
             (weights, train_n, holdout_spearman, holdout_p20, promoted, note)
           VALUES (?,?,?,?,?,?)""",
        (json.dumps(new_weights), len(train_f), hold_sp_new, hold_p20,
         int(promoted), note),
    )
    conn.commit()
    version = cur.lastrowid

    if promoted:
        score_all(conn, cfg, weights=new_weights, version=version)

    return {
        "promoted": promoted,
        "note": note,
        "version": version,
        "train_n": len(train_f),
        "holdout_n": len(hold_f),
        "train_spearman": round(train_sp, 3),
        "holdout_spearman": None if hold_sp_new is None else round(hold_sp_new, 3),
        "holdout_p20": None if hold_p20 is None else round(hold_p20, 3),
        "old_weights": {k: round(v, 3) for k, v in prev_weights.items()},
        "new_weights": {k: round(v, 3) for k, v in new_weights.items()},
    }
