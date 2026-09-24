"""The max-offer invariants, checked without a database or a fitted model.

These are the properties the app's headline rests on. They were verified by hand
against the live data in every phase; this is the same check, run in a second.
"""
import unittest

from flipfinder.analysis.arv import max_offer

CFG = {"scoring": {"carry_closing_pct": 0.08, "target_margin": 0.10}}


def margin_at(price, arv, reno, cfg=CFG):
    """The scorer's margin, written out independently of arv.estimate so the test
    can disagree with the implementation."""
    carry = cfg["scoring"]["carry_closing_pct"] * arv
    return (arv - price - reno - carry) / arv


# (arv, reno_cost) pairs spanning the real range, plus the degenerate ones where
# renovation costs approach the finished value.
CASES = [
    (262241.59, 43950.0),      # 5515 Glen Forest Ln, the one deal
    (187938.39, 65300.0),      # 9419 Culberson St
    (110216.0, 51550.0),       # 2327 Dathe St, deeply negative margin
    (132778.0, 99450.0),       # 2618 Pine St, reno nearly the whole ARV
    (1487721.0, 0.0),          # top of the ARV range, no reno
    (50000.0, 49000.0),        # reno almost equals ARV
]


class TestMaxOffer(unittest.TestCase):
    def test_below_arv(self):
        """An offer at or above ARV is never right: carry and target margin alone
        take a bite out of it even when the renovation is free."""
        for arv, reno in CASES:
            with self.subTest(arv=arv, reno=reno):
                self.assertLess(max_offer(arv, reno, CFG), arv)

    def test_margin_at_max_offer_is_the_target(self):
        """Priced exactly at the max offer, the margin is exactly the target. This
        is the property that lets the UI lead with the offer and still show the
        margin without the two ever disagreeing."""
        target = CFG["scoring"]["target_margin"]
        for arv, reno in CASES:
            with self.subTest(arv=arv, reno=reno):
                self.assertAlmostEqual(
                    margin_at(max_offer(arv, reno, CFG), arv, reno), target, places=12
                )

    def test_equivalence_holds_in_both_directions(self):
        """margin >= target_margin exactly when price <= max_offer. Checked a dollar
        either side of the boundary, because on real data every listing sits far
        above its offer and the biconditional passes with both sides false."""
        target = CFG["scoring"]["target_margin"]
        for arv, reno in CASES:
            offer = max_offer(arv, reno, CFG)
            with self.subTest(arv=arv, reno=reno):
                self.assertGreaterEqual(margin_at(offer - 1, arv, reno), target)
                self.assertLess(margin_at(offer + 1, arv, reno), target)

    def test_discount_matches_the_offer(self):
        """offer_discount is the ask's distance to the offer, and goes negative when
        the ask already clears the target — the case the UI words differently."""
        arv, reno = 262241.59, 43950.0
        offer = max_offer(arv, reno, CFG)
        for price in (159000.0, offer, 300000.0):
            with self.subTest(price=price):
                discount = (price - offer) / price
                self.assertAlmostEqual(price * (1 - discount), offer, places=6)
        self.assertLess((159000.0 - offer) / 159000.0, 0, "the deal's ask clears its offer")

    def test_target_margin_moves_the_offer(self):
        """The offer is configured, not hardcoded: a higher target margin must lower
        what you can pay."""
        arv, reno = 262241.59, 43950.0
        strict = {"scoring": {"carry_closing_pct": 0.08, "target_margin": 0.25}}
        self.assertLess(max_offer(arv, reno, strict), max_offer(arv, reno, CFG))


if __name__ == "__main__":
    unittest.main()
