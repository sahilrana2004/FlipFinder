import argparse
import json

from flipfinder.analysis import backtest, distress, fitter, photos, scorer
from flipfinder.config import load_config
from flipfinder.db import init_db
from flipfinder.ingest import census, redfin, redfin_detail


BROWSER_HINT = (
    "Redfin's WAF blocked the request (it fingerprints non-browser TLS).\n"
    "  Use the browser path instead: open https://www.redfin.com, paste\n"
    "  scripts/redfin_fetch.js into the DevTools console, move the three\n"
    "  downloaded files into FlipFinder/data, then run: py run.py import"
)


def cmd_ingest(conn, cfg):
    import requests

    try:
        for name, n in redfin.ingest_listings(conn, cfg):
            print(f"[ingest] {n} active listings for {name}")
        days = cfg["market"]["sold_within_days"]
        for name, (kept, skipped) in redfin.ingest_solds(conn, cfg):
            print(f"[ingest] {kept} solds ({days}d, {skipped} skipped) for {name}")
    except requests.HTTPError as e:
        print(f"[ingest] {e}\n{BROWSER_HINT}")
        raise SystemExit(1)


def cmd_import(conn, _cfg):
    """Import data/{active,sold}.csv + data/enrich.json produced by
    scripts/redfin_fetch.js (Redfin's WAF blocks non-browser TLS fingerprints)."""
    import csv
    from pathlib import Path

    data = Path(__file__).parent / "data"
    redfin.deactivate_all(conn)
    for pattern, fn in (("active*.csv", redfin.upsert_active_rows),
                        ("sold*.csv", redfin.upsert_sold_rows)):
        paths = sorted(data.glob(pattern))
        if not paths:
            print(f"[import] no {pattern} in {data} — skipped")
            continue
        for p in paths:
            with open(p, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            result = fn(conn, rows)
            if pattern == "sold*.csv":
                kept, skipped = result
                print(f"[import] {p.name}: {kept} kept, {skipped} skipped")
            else:
                print(f"[import] {p.name}: {result} rows")

    p = data / "enrich.json"
    if not p.exists():
        print(f"[import] {p} missing — skipped")
        return
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    n = 0
    for item in items:
        row = conn.execute(
            "SELECT id FROM listings WHERE url=?", (item["url"],)
        ).fetchone()
        if not row:
            continue
        conn.execute(
            "UPDATE listings SET remarks=? WHERE id=?", (item["remarks"], row["id"])
        )
        for url in item.get("photos", [])[:8]:
            conn.execute(
                "INSERT OR IGNORE INTO photos (listing_id, url) VALUES (?,?)",
                (row["id"], url),
            )
        n += 1
    conn.commit()
    print(f"[import] enrich.json: {n} listings")


def cmd_enrich(conn, cfg):
    done, failed = redfin_detail.enrich(conn, cfg)
    print(f"[enrich] remarks+photos: {done} ok, {failed} failed")
    scored, note = photos.score_all(conn, cfg)
    print(f"[photos] condition scored: {scored} ({note})")


def cmd_census(conn, cfg):
    n = census.assign_tracts(conn)
    print(f"[census] tracts assigned: {n}")
    n = census.fetch_acs(conn, cfg)
    print(f"[census] ACS tract stats: {n}")
    n = census.refresh_market_stats(conn)
    print(f"[census] market stats refreshed for {n} tracts")


def cmd_score(conn, cfg):
    n = distress.compute(conn, cfg)
    print(f"[distress] computed for {n} listings")
    n, version = scorer.score_all(conn, cfg)
    print(f"[score] {n} listings scored (scorer v{version})")


def cmd_fit(conn, cfg):
    print(json.dumps(fitter.run(conn, cfg), indent=2))


def cmd_backtest(conn, cfg):
    result = backtest.run(conn, cfg)
    print(f"[backtest] {result['n_targets']} targets "
          f"({result['n_tune']} tune / {result['n_test']} test)")
    print(f"[backtest] leakage check: {result['leakage_checked']} checked, "
          f"{result['leakage_violations']} violations")
    print("[backtest] wrote data/backtest/{metrics.json,errors.csv,summary.txt}")


def cmd_serve(_conn, cfg):
    import uvicorn

    s = cfg["server"]
    uvicorn.run("flipfinder.app.server:app", host=s["host"], port=s["port"])


COMMANDS = {
    "ingest": cmd_ingest,
    "import": cmd_import,
    "enrich": cmd_enrich,
    "census": cmd_census,
    "score": cmd_score,
    "fit": cmd_fit,
    "backtest": cmd_backtest,
    "serve": cmd_serve,
}


def cmd_pipeline(conn, cfg):
    """Import browser-fetched data, then enrich and score it."""
    for step in ("import", "census", "score"):
        COMMANDS[step](conn, cfg)


COMMANDS["pipeline"] = cmd_pipeline


def main():
    parser = argparse.ArgumentParser(prog="flipfinder")
    parser.add_argument("command", choices=list(COMMANDS))
    args = parser.parse_args()
    cfg = load_config()
    conn = init_db()
    COMMANDS[args.command](conn, cfg)


if __name__ == "__main__":
    main()
