"""A fabricated dataset so the app runs with no Redfin access.

Nothing here is scraped. Addresses, cities, ZIPs, remarks and prices are invented
from a fixed seed, and every listing says so in its remarks. The coordinates are a
grid over Dallas only so the map has somewhere to draw; they don't correspond to
the addresses, which don't exist.

What this demonstrates is the pipeline and the interface, not the model's accuracy.
The AVM fitted here is fitted on invented sales with fixed parameters, so its error
on this data says nothing about its error on real sales. The measured numbers are
in the README, and they come from the backtest over real sold data.
"""
import random
from datetime import date, timedelta

SEED = 20260921
CITY = "Sampleton"
STATE = "TX"

# Deliberately impossible ZIPs, so nothing here can be mistaken for a real market.
# base_ppsf is what a house in this fake tract closes for per square foot.
TRACTS = [
    {"tract": "99000000001", "zip": "00001", "base_ppsf": 96.0, "lat": 32.74, "lng": -96.86},
    {"tract": "99000000002", "zip": "00002", "base_ppsf": 118.0, "lat": 32.71, "lng": -96.79},
    {"tract": "99000000003", "zip": "00003", "base_ppsf": 141.0, "lat": 32.78, "lng": -96.75},
    {"tract": "99000000004", "zip": "00004", "base_ppsf": 163.0, "lat": 32.82, "lng": -96.83},
    {"tract": "99000000005", "zip": "00005", "base_ppsf": 187.0, "lat": 32.69, "lng": -96.72},
]

STREETS = [
    "Example St", "Sample Ave", "Placeholder Dr", "Fixture Ln", "Stub Ct",
    "Mock Blvd", "Dummy Way", "Fake Rd", "Test Trl", "Invented Cir",
]

PREFIX = "SYNTHETIC DEMO DATA, not a real listing."
DISTRESSED_REMARKS = [
    "Sold as-is, cash only. Needs work throughout; bring your vision.",
    "Investor special. Great bones, needs updating top to bottom.",
    "Handyman opportunity. Estate sale, no repairs will be made.",
    "Fixer with real upside. TLC needed; sold as is, where is.",
    "Diamond in the rough. Rehab project for the right cash buyer.",
]
RENOVATED_REMARKS = [
    "Fully updated with new paint, granite counters and stainless appliances.",
    "Move-in ready. Updated kitchen, quartz counters, new roof.",
    "Beautifully renovated throughout with new flooring and fixtures.",
    "Remodeled kitchen and baths, stainless appliances, updated systems.",
]
PLAIN_REMARKS = [
    "Well kept three bedroom on a quiet street. Large back yard.",
    "Single story with an open plan and a two car garage.",
    "Comfortable family home close to schools and shopping.",
    "Solid home with mature trees and a covered patio.",
]


def _address(rng, used):
    while True:
        a = f"{rng.randrange(100, 9999)} {rng.choice(STREETS)}"
        if a not in used:
            used.add(a)
            return a


def _house(rng, tract):
    sqft = rng.randrange(800, 2500, 10)
    beds = 2 if sqft < 1100 else (3 if sqft < 1900 else 4)
    return {
        "sqft": float(sqft),
        "beds": float(beds),
        "baths": float(rng.choice([1, 2, 2, 2, 3])),
        "year_built": rng.randrange(1948, 2006),
        "lot_sqft": float(rng.randrange(5000, 11000, 100)),
        "lat": tract["lat"] + rng.uniform(-0.012, 0.012),
        "lng": tract["lng"] + rng.uniform(-0.012, 0.012),
    }


def _condition(rng):
    """(value multiplier, remarks pool). Distressed houses close below the
    tract's rate, renovated ones above it - the spread the scorer looks for."""
    roll = rng.random()
    if roll < 0.22:
        return rng.uniform(0.62, 0.80), DISTRESSED_REMARKS
    if roll < 0.48:
        return rng.uniform(1.06, 1.22), RENOVATED_REMARKS
    return rng.uniform(0.92, 1.05), PLAIN_REMARKS


def _has_real_data(conn):
    """Anything whose URL points at a real site is a real pull, not a demo build."""
    return conn.execute(
        "SELECT COUNT(*) c FROM listings WHERE url IS NOT NULL AND url NOT LIKE 'demo://%'"
    ).fetchone()["c"] or conn.execute(
        "SELECT COUNT(*) c FROM sold WHERE url IS NOT NULL AND url NOT LIKE 'demo://%'"
    ).fetchone()["c"]


def build(conn, cfg, n_sold=700, n_active=110):
    """Replace the demo dataset and fit an ARV model on it. Returns a summary."""
    if _has_real_data(conn):
        raise SystemExit(
            "[demo] this database holds real scraped data - refusing to overwrite it.\n"
            "  The demo dataset is for a clean checkout. To build one anyway, move\n"
            "  flipfinder.db aside first and re-run; nothing is deleted either way."
        )
    rng = random.Random(SEED)
    today = date.today()
    used = set()

    conn.execute("DELETE FROM sold WHERE url LIKE 'demo://%'")
    conn.execute("DELETE FROM listings WHERE url LIKE 'demo://%'")
    conn.execute("DELETE FROM scores")
    conn.execute("DELETE FROM area_stats")
    conn.commit()

    for i in range(n_sold):
        t = rng.choice(TRACTS)
        h = _house(rng, t)
        mult, pool = _condition(rng)
        sold_date = today - timedelta(days=rng.randrange(20, 1080))
        # Prices drift up ~4%/year, so the point-in-time features have a trend to find.
        drift = 1.04 ** ((sold_date - (today - timedelta(days=1095))).days / 365.0)
        close_ppsf = t["base_ppsf"] * mult * drift * rng.uniform(0.93, 1.07)
        close_price = round(close_ppsf * h["sqft"], -2)
        list_price = round(close_price * rng.uniform(1.0, 1.09), -2)
        conn.execute(
            """INSERT INTO sold (url, address, city, zip, price, sold_date, beds, baths,
                                 sqft, year_built, ppsf, lat, lng, tract, list_price,
                                 list_price_source, close_price, close_price_source,
                                 close_ppsf, remarks, lot_sqft, dom)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"demo://sold/{i}", _address(rng, used), CITY, t["zip"], list_price,
             sold_date.isoformat(), h["beds"], h["baths"], h["sqft"], h["year_built"],
             list_price / h["sqft"], h["lat"], h["lng"], t["tract"], list_price,
             "demo", close_price, "demo", close_price / h["sqft"],
             f"{PREFIX} {rng.choice(pool)}", h["lot_sqft"], None),
        )

    for i in range(n_active):
        t = rng.choice(TRACTS)
        h = _house(rng, t)
        mult, pool = _condition(rng)
        value = t["base_ppsf"] * mult * 1.12 * h["sqft"]
        # A handful are asked well under what the tract supports. Without them the
        # demo would be a list of passes, which shows the scoring but not the point.
        ask_ratio = rng.uniform(0.58, 0.74) if i < 12 else rng.uniform(0.92, 1.18)
        price = round(value * ask_ratio, -2)
        if not 30000 <= price <= 400000:
            price = round(min(max(price, 45000), 390000), -2)
        conn.execute(
            """INSERT INTO listings (url, address, city, state, zip, price, beds, baths,
                                     sqft, lot_sqft, year_built, dom, ppsf, lat, lng,
                                     status, remarks, tract, active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (f"demo://listing/{i}", _address(rng, used), CITY, STATE, t["zip"], price,
             h["beds"], h["baths"], h["sqft"], h["lot_sqft"], h["year_built"],
             rng.randrange(2, 140), price / h["sqft"], h["lat"], h["lng"], "Active",
             f"{PREFIX} {rng.choice(pool)}", t["tract"]),
        )
    conn.commit()
    return {"sold": n_sold, "active": n_active, "tracts": len(TRACTS)}


# Fixed, not searched. The real parameters come from the backtest's tune split
# (see data/backtest/metrics.json); on 700 invented sales a search would be fitting
# noise, and the demo model exists to make the UI render, not to be accurate.
FIT_PARAMS = {"objective": "huber", "num_leaves": 15, "min_data_in_leaf": 20,
              "rounds": 300, "low_weight": 6.0}


def fit_model(conn):
    from .analysis import avm

    records = avm.training_records(conn)
    if not records:
        raise SystemExit("[demo] no synthetic sales to fit on - run `py run.py demo` first")
    booster, meta = avm.fit_records(records, FIT_PARAMS)
    meta.update(candidate="demo-synthetic", trained_on=len(records),
                newest_sale=records[-1]["sold_date"])
    avm.save(booster, meta)
    avm._cache.clear()
    return meta
