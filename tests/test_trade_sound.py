from agent_stonks.trade_sound import next_trade_cue


class _D:
    """Minimal stand-in for a Decision -- the cue logic reads three fields."""

    def __init__(self, action="buy", status="filled", ts="2026-09-11T14:00:00Z"):
        self.action = action
        self.status = status
        self.ts = ts


class TestNextTradeCue:
    def test_first_look_adopts_history_silently(self):
        # Opening the tab on a session that already traded must not fire a
        # chime per historical trade.
        cue, seen = next_trade_cue([_D(), _D(), _D()], None)
        assert cue is None
        assert seen == 3

    def test_nothing_new_makes_no_sound(self):
        cue, seen = next_trade_cue([_D(), _D()], 2)
        assert cue is None
        assert seen == 2

    def test_a_new_fill_produces_a_cue(self):
        cue, seen = next_trade_cue([_D(), _D(ts="2026-09-11T14:05:00Z")], 1)
        assert cue is not None
        assert cue["side"] == "buy"
        assert seen == 2

    def test_a_sell_is_distinguishable_from_a_buy(self):
        # The whole point of the cue is telling what happened without looking.
        cue, _ = next_trade_cue([_D(action="sell")], 0)
        assert cue["side"] == "sell"

    def test_several_fills_in_one_poll_window_sound_once(self):
        # Chiming per trade would overlap into noise; the newest one speaks.
        cue, seen = next_trade_cue(
            [_D(), _D(), _D(action="sell", ts="2026-09-11T14:09:00Z")], 0
        )
        assert cue["side"] == "sell"
        assert seen == 3

    def test_ignores_decisions_that_are_not_trades(self):
        decisions = [
            _D(action="alert", status="noop"),
            _D(action="tactics", status="armed"),
            _D(action="sleep", status="noop"),
        ]
        cue, seen = next_trade_cue(decisions, 0)
        assert cue is None
        assert seen == 0

    def test_ignores_a_rejected_order(self):
        # A refused order is not a trade; the broker can reject plenty of them.
        cue, seen = next_trade_cue([_D(status="rejected")], 0)
        assert cue is None
        assert seen == 0

    def test_counts_only_fills_so_a_rejection_between_them_does_not_shift_it(self):
        decisions = [_D(), _D(status="rejected"), _D(action="sell")]
        cue, seen = next_trade_cue(decisions, 1)
        assert cue["side"] == "sell"
        assert seen == 2

    def test_a_replaced_tracker_resets_rather_than_going_negative(self):
        # Starting a new agent run swaps in a fresh ledger with fewer trades;
        # that must not produce a negative count or a spurious chime.
        cue, seen = next_trade_cue([_D()], 7)
        assert cue is None
        assert seen == 1

    def test_an_empty_ledger_is_quiet(self):
        assert next_trade_cue([], None) == (None, 0)
        assert next_trade_cue([], 0) == (None, 0)

    def test_cue_ids_differ_between_consecutive_trades(self):
        # The frontend suppresses a repeat of the same id, so two trades that
        # share a timestamp must still get distinct ids or the second is silent.
        same_ts = "2026-09-11T14:00:00Z"
        first, seen = next_trade_cue([_D(ts=same_ts)], 0)
        second, _ = next_trade_cue([_D(ts=same_ts), _D(ts=same_ts)], seen)
        assert first["id"] != second["id"]
