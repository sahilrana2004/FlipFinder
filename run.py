import argparse
import json

from flipfinder.analysis import avm, backtest, distress, fitter, photos, scorer
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
            print(f"[import] no {pattern} in {data} - skipped")
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

    details = sorted(data.glob("sold_detail_*.json"))
    for p in details:
        with open(p, encoding="utf-8") as f:
            n = redfin_detail.import_sold_details(conn, json.load(f))
        print(f"[import] {p.name}: {n} sales updated")

    # enrich*.json, not enrich.json: a pull that covers new ZIPs only would otherwise
    # have to overwrite the remarks and photos of every listing pulled before it.
    enrichments = sorted(data.glob("enrich*.json"))
    if not enrichments:
        print(f"[import] no enrich*.json in {data} - skipped")
        return
    for p in enrichments:
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
        print(f"[import] {p.name}: {n} listings")


def cmd_enrich(conn, cfg):
    done, failed = redfin_detail.enrich(conn, cfg)
    print(f"[enrich] remarks+photos: {done} ok, {failed} failed")
    print("[enrich] AI labeling needs scores: run `py run.py refresh`")


def _label(conn, cfg):
    """Label every eligible listing. Returns (labeled, failed, no_photos). Ollama is
    only required when something is eligible, so a refresh with nothing new never
    touches the model."""
    no_photos = conn.execute(
        "SELECT COUNT(*) c FROM scores s WHERE NOT EXISTS "
        "(SELECT 1 FROM photos p WHERE p.listing_id = s.listing_id)"
    ).fetchone()["c"]
    cap = cfg["ollama"]["max_labels_per_run"]
    waiting = len(photos.eligible(conn, cfg))
    pending = min(waiting, cap)
    deferred = waiting - pending
    print(f'[ailabel] model {cfg["ollama"]["model"]}, prompt {photos.PROMPT_VERSION}; '
          f"{pending} to label, {deferred} left for the next run (cap {cap}), "
          f"{no_photos} scored listings skipped (no photos)")
    if not pending:
        return 0, 0, no_photos
    if not photos.ollama_available(cfg):
        print(f'[ailabel] Ollama not reachable at {cfg["ollama"]["url"]}')
        raise SystemExit(1)
    labeled = failed = 0
    elapsed = 0.0
    for i, total, listing, result in photos.label_all(conn, cfg, limit=cap):
        if result is None:
            failed += 1
            print(f"[ailabel] {i}/{total} {listing['address']}: FAILED (see {photos.FAILURE_LOG})")
            continue
        labeled += 1
        elapsed += result["seconds"]
        eta = elapsed / labeled * (total - i)
        flags = " downgrade" if result["downgrade"] else ""
        flags += " text_conflict" if result["text_conflict"] else ""
        print(f"[ailabel] {i}/{total} {listing['address']}: {result['seconds']:.0f}s "
              f"condition {result['condition']} {result['reno_scope']}{flags} "
              f"(ETA {eta / 60:.1f} min)")
    print(f"[ailabel] {labeled} labeled, {failed} failed")
    return labeled, failed, no_photos


def _refresh_verdicts(conn, cfg):
    n = photos.refresh_verdicts(conn, cfg)
    print(f"[ailabel] verdicts refreshed from current margins: {n}")


def cmd_ailabel(conn, cfg):
    labeled, _, _ = _label(conn, cfg)
    if labeled:
        cmd_score(conn, cfg)
    _refresh_verdicts(conn, cfg)


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
    for name, m in result["primary"].items():
        lo, hi = m["within_20pct_ci"]
        w20 = "n/a" if m["within_20pct"] is None else f"{m['within_20pct'] * 100:.1f}%"
        ci = "" if lo is None else f" [{lo * 100:.1f}, {hi * 100:.1f}]"
        print(f"[backtest] test, listed <= $145k, {name}: n={m['n_predicted']} "
              f"within 20% {w20}{ci}")
    print("[backtest] wrote data/backtest/{metrics.json,errors.csv,summary.txt}")


def cmd_train(conn, _cfg):
    meta = avm.train(conn)
    print(f"[train] ARV model ({meta['candidate']}) fit on {meta['trained_on']} sales "
          f"through {meta['newest_sale']}; saved to {avm.MODEL_PATH}")


def cmd_serve(_conn, cfg):
    import uvicorn

    s = cfg["server"]
    uvicorn.run("flipfinder.app.server:app", host=s["host"], port=s["port"])


def cmd_demo(conn, cfg):
    """Fabricated data so the app runs with no Redfin access."""
    from flipfinder import demo

    built = demo.build(conn, cfg)
    print(f"[demo] {built['sold']} synthetic sales and {built['active']} synthetic listings "
          f"across {built['tracts']} invented tracts (nothing scraped)")
    meta = demo.fit_model(conn)
    print(f"[demo] ARV model fit on {meta['trained_on']} synthetic sales -> {avm.MODEL_PATH}")
    n = census.refresh_market_stats(conn)
    print(f"[demo] market stats for {n} tracts")
    cmd_score(conn, cfg)
    print("[demo] ready. Now run: py run.py serve")


# name -> (handler, one-line description for --help)
COMMANDS = {
    "demo": (cmd_demo, "Build a fabricated dataset and model so the app runs with no data"),
    "import": (cmd_import, "Load data/{active,sold}*.csv and enrich*.json from the browser fetch"),
    "ingest": (cmd_ingest, "Pull listings over HTTP (blocked by Redfin's WAF; use the browser script)"),
    "enrich": (cmd_enrich, "Fetch remarks and photos for listings that are missing them"),
    "census": (cmd_census, "Assign census tracts and rebuild per-tract market stats"),
    "score": (cmd_score, "Score every active listing in the buy box"),
    "ailabel": (cmd_ailabel, "Read listing photos with the local vision model"),
    "backtest": (cmd_backtest, "Measure the AVM against held-out sales and pick its config"),
    "train": (cmd_train, "Refit the ARV model on every sale, using the backtest's config"),
    "fit": (cmd_fit, "Refit the scoring weights from your saved labels"),
    "serve": (cmd_serve, "Start the web UI"),
}


def cmd_refresh(conn, cfg):
    """Import browser-fetched data, score it, AI-label anything new, then re-score
    so the labels' reno scopes flow into margins and verdicts."""
    cmd_import(conn, cfg)
    cmd_census(conn, cfg)
    cmd_score(conn, cfg)
    labeled, failed, no_photos = _label(conn, cfg)
    cmd_score(conn, cfg)
    _refresh_verdicts(conn, cfg)

    active = conn.execute("SELECT COUNT(*) c FROM listings WHERE active=1").fetchone()["c"]
    scored = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]
    counts = {name: 0 for name in photos.VERDICT_NAMES}
    for r in conn.execute(
        """SELECT a.suggested_verdict FROM ai_labels a
           JOIN scores s ON s.listing_id = a.listing_id
           WHERE a.model = ? AND a.prompt_version = ?""",
        (cfg["ollama"]["model"], photos.PROMPT_VERSION),
    ):
        counts[photos.VERDICT_NAMES[r["suggested_verdict"]]] += 1
    print(f"[refresh] active {active}, scored {scored}, labeled this run {labeled}, "
          f"skipped (no photos) {no_photos}, failed {failed}")
    print(f"[refresh] AI verdicts: deal {counts['deal']}, maybe {counts['maybe']}, "
          f"pass {counts['pass']}")


COMMANDS["refresh"] = (cmd_refresh, "import -> census -> score -> label -> score (the usual run)")
COMMANDS["pipeline"] = (cmd_refresh, "Alias for refresh")


def main():
    parser = argparse.ArgumentParser(
        prog="flipfinder",
        description="Find flip candidates: ingest listings, price them, score them, serve them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:" + "".join(
            chr(10) + f"  {name:<10} {help_text}"
            for name, (_, help_text) in COMMANDS.items()
        ),
    )
    parser.add_argument("command", choices=list(COMMANDS), metavar="command",
                        help="one of the commands listed below")
    args = parser.parse_args()
    cfg = load_config()
    conn = init_db()
    try:
        COMMANDS[args.command][0](conn, cfg)
    except FileNotFoundError as e:
        # A missing model or missing backtest output is an ordinary "you skipped a
        # step" on a fresh clone. The message already says which step; a traceback
        # on top of it just buries the instruction.
        raise SystemExit(f"[{args.command}] {e}")


if __name__ == "__main__":
    main()
