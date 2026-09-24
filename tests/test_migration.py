"""Schema migration behaviour on a database that predates the current columns.

The specific regression this guards: scores is a derived cache, and a row written
before max_offer existed has no way to acquire one. Left in place it renders as
"No offer -- ARV unavailable" on every listing, which looks like a broken deploy.
It also reads back as a NULL column, which silently answers "0 listings within 15%
of ask" to a question nobody meant to ask about NULLs.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from flipfinder import db


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "t.db"
        self._real = db.DB_PATH
        db.DB_PATH = self.path

    def tearDown(self):
        db.DB_PATH = self._real
        self._tmp.cleanup()

    def _columns(self, conn, table):
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}

    def test_adds_columns_to_an_old_database(self):
        conn = db.init_db()
        conn.execute("ALTER TABLE scores DROP COLUMN max_offer")
        conn.execute("ALTER TABLE scores DROP COLUMN offer_discount")
        conn.commit()
        conn.close()

        conn = db.init_db()
        self.assertIn("max_offer", self._columns(conn, "scores"))
        self.assertIn("offer_discount", self._columns(conn, "scores"))
        conn.close()

    def test_stale_scores_are_dropped_not_left_half_filled(self):
        conn = db.init_db()
        conn.execute("ALTER TABLE scores DROP COLUMN max_offer")
        conn.execute("ALTER TABLE scores DROP COLUMN offer_discount")
        conn.execute(
            "INSERT INTO scores (listing_id, scorer_version, score, margin, arv, "
            "reno_cost, spread, components) VALUES (1, 0, 0.5, -0.1, 200000, 50000, "
            "-20000, '{}')"
        )
        conn.commit()
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"], 1)
        conn.close()

        conn = db.init_db()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"], 0,
            "a score row that predates max_offer must be dropped, not served with a NULL",
        )
        conn.close()

    def test_migration_does_not_touch_source_data(self):
        """scores is a cache and may be discarded. listings and sold are not."""
        conn = db.init_db()
        conn.execute("ALTER TABLE scores DROP COLUMN max_offer")
        conn.execute("INSERT INTO listings (url, address, price) VALUES ('u1', 'a', 1)")
        conn.execute("INSERT INTO sold (url, address, price) VALUES ('u2', 'b', 2)")
        conn.commit()
        conn.close()

        conn = db.init_db()
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM listings").fetchone()["c"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) c FROM sold").fetchone()["c"], 1)
        conn.close()

    def test_migration_is_idempotent(self):
        conn = db.init_db()
        conn.execute(
            "INSERT INTO scores (listing_id, scorer_version, score, max_offer, "
            "offer_discount, components) VALUES (1, 0, 0.5, 120000, 0.2, '{}')"
        )
        conn.commit()
        conn.close()

        # Re-running init_db on an up-to-date schema must not wipe fresh scores.
        conn = db.init_db()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"], 1,
            "the backfill must fire only when the column is first added",
        )
        conn.close()


if __name__ == "__main__":
    unittest.main()
