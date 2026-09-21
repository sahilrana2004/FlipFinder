import json
import sqlite3

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  mls TEXT,
  source TEXT DEFAULT 'redfin',
  url TEXT UNIQUE,
  address TEXT, city TEXT, state TEXT, zip TEXT,
  price REAL, beds REAL, baths REAL, sqft REAL, lot_sqft REAL,
  year_built INTEGER, dom INTEGER, ppsf REAL,
  lat REAL, lng REAL,
  status TEXT,
  remarks TEXT,
  tract TEXT,
  condition_score REAL,
  distress_score REAL,
  distress_signals TEXT,
  first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
  last_seen TEXT DEFAULT CURRENT_TIMESTAMP,
  active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sold (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE,
  address TEXT, city TEXT, zip TEXT,
  price REAL, sold_date TEXT,
  beds REAL, baths REAL, sqft REAL,
  year_built INTEGER, ppsf REAL,
  lat REAL, lng REAL,
  tract TEXT
);

CREATE TABLE IF NOT EXISTS photos (
  listing_id INTEGER,
  url TEXT,
  PRIMARY KEY (listing_id, url)
);

CREATE TABLE IF NOT EXISTS area_stats (
  tract TEXT PRIMARY KEY,
  median_income REAL,
  median_home_value REAL,
  vacancy_rate REAL,
  sold_count REAL,
  active_count REAL,
  median_sold_ppsf REAL,
  p75_sold_ppsf REAL,
  std_sold_ppsf REAL,
  median_dom REAL,
  updated TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
  listing_id INTEGER PRIMARY KEY,
  scorer_version INTEGER,
  score REAL,
  raw_score REAL,
  margin REAL, arv REAL, reno_cost REAL, spread REAL,
  confidence REAL, liquidity REAL, distress REAL, size_mult REAL,
  components TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  listing_id INTEGER UNIQUE,
  verdict INTEGER,              -- 0=pass 1=maybe 2=deal
  tags TEXT,
  feature_snapshot TEXT,        -- feature vector frozen at rating time
  algo_score_shown REAL,        -- NULL when rated blind
  scorer_version INTEGER,
  split TEXT,                   -- 'train' | 'holdout', assigned at insert
  ts TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scorer_versions (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  weights TEXT,
  train_n INTEGER,
  holdout_spearman REAL,
  holdout_p20 REAL,
  promoted INTEGER DEFAULT 0,
  note TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Model suggestions only. Human labels in `labels` stay the sole ground truth for
-- fitting and metrics; training on these would teach the scorer the model.
CREATE TABLE IF NOT EXISTS ai_labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  listing_id INTEGER,
  model TEXT,
  prompt_version TEXT,
  condition INTEGER,            -- 1-10, what the photos show
  reno_scope TEXT,              -- light | medium | gut
  red_flags TEXT,               -- JSON array
  photos_representative INTEGER,
  text_conflict INTEGER,        -- remarks describe a materially different condition
  suggested_verdict INTEGER,    -- 0=pass 1=maybe 2=deal; from v4, derived from margin + downgrade
  reasons TEXT,                 -- JSON array, <= 3
  notes TEXT,
  photo_count INTEGER,
  seconds REAL,
  raw_response TEXT,
  ts TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (listing_id, model, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_sold_tract ON sold(tract);
CREATE INDEX IF NOT EXISTS idx_listings_tract ON listings(tract);
"""


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


# CREATE TABLE IF NOT EXISTS never alters an existing table, so columns added after
# a DB was created are applied here, only when missing.
ADDED_COLUMNS = {
    "labels": [
        ("ai_label_id", "INTEGER"),
        ("ai_shown", "INTEGER DEFAULT 0"),   # anchoring: was the AI suggestion visible
        ("condition_human", "INTEGER"),
    ],
    "listings": [
        ("condition_source", "TEXT"),        # 'ai' | 'human' | NULL
        ("reno_scope", "TEXT"),              # light | medium | gut, from the AI label
    ],
    "ai_labels": [
        ("downgrade", "INTEGER"),            # v4+: one step below the margin ceiling
    ],
    # The headline the app leads with, alongside the margin it is derived from.
    # score_all rewrites every row each run, so these need no backfill.
    "scores": [
        ("max_offer", "REAL"),               # highest price that still clears target_margin
        ("offer_discount", "REAL"),          # (price - max_offer) / price
    ],
    "area_stats": [
        # sold_count spans every sale held for the tract (3 years), which compared
        # against a current active count reads as boundless liquidity. Liquidity uses
        # this trailing-12-month count instead.
        ("sold_count_12m", "REAL"),
    ],
    # Texas doesn't disclose sale prices: the CSV's sold PRICE is the last asking
    # price. The real close price comes from the detail page's MLS ratio fields.
    "sold": [
        ("list_price", "REAL"),              # last asking price before the sale
        ("list_price_source", "TEXT"),       # how list_price was extracted
        ("close_price", "REAL"),             # what it actually closed for
        ("close_price_source", "TEXT"),
        ("remarks", "TEXT"),                 # agent remarks from the sold listing
        ("lot_sqft", "REAL"),
        ("dom", "INTEGER"),
        ("detail_fetched", "TEXT"),          # set once the detail page was scraped
        # sold.ppsf is price/sqft, i.e. LIST dollars. Every comp, tract median and
        # market stat is built from close_ppsf instead, so nothing mixes currencies.
        ("close_ppsf", "REAL"),
    ],
}

# Run once, when the column above is first added, so an existing DB gets the value
# for every sale whose close price is already recovered.
BACKFILL = {
    ("sold", "close_ppsf"):
        "UPDATE sold SET close_ppsf = close_price / sqft "
        "WHERE close_price IS NOT NULL AND sqft > 0",
}


def _migrate(conn):
    for table, cols in ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                sql = BACKFILL.get((table, name))
                if sql:
                    conn.execute(sql)


def init_db():
    conn = connect()
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def current_weights(conn, cfg):
    """Latest promoted fitted weights, falling back to config defaults."""
    row = conn.execute(
        "SELECT version, weights FROM scorer_versions WHERE promoted=1 "
        "ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row:
        return row["version"], json.loads(row["weights"])
    return 0, dict(cfg["scoring"]["weights"])
