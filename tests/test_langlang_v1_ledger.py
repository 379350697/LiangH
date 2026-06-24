import json
import os
import tempfile
import unittest

from langlang_trader.ledger import Ledger
from langlang_trader.models import EntrySetup, LangLangSignal, MarketRegime, Side


class LangLangV1LedgerTest(unittest.TestCase):
    def test_signal_records_strategy_state_and_explanation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(os.path.join(tmp, "v1.sqlite3"), run_id="run-v1", bot_id="bot-v1", variant_id="variant-v1")
            signal = LangLangSignal(
                symbol="BTC-USDT-SWAP",
                side=Side.LONG,
                strength=0.78,
                reason_codes=["daily_main_uptrend", "intraday_reclaim_confirmed"],
                filter_codes=["no_failure_filter"],
                features={"ret_20d": 0.28},
                invalidation_price=95.0,
                stop_loss=95.0,
                take_profit_hint=130.0,
                take_profit_plan={"partial_r": 2.0},
                hold_plan={"runner": True},
                strategy_version="rules_langlang_v1",
                regime=MarketRegime.STRONG_PULLBACK,
                setup=EntrySetup.FIRST_PULLBACK,
                decision_trace={"action": "enter", "explanation": "unit-test"},
                historical_match_score=0.62,
                created_at="2024-03-12T00:00:00+00:00",
            )

            signal_id = ledger.record_signal(signal, strategy_version=signal.strategy_version)

            rows = ledger.list_rows("signals")
            self.assertEqual(signal_id, rows[0]["id"])
            self.assertEqual(rows[0]["strategy_version"], "rules_langlang_v1")
            self.assertEqual(rows[0]["regime"], "strong_pullback")
            self.assertEqual(rows[0]["setup"], "first_pullback")
            self.assertEqual(json.loads(rows[0]["filter_codes_json"]), ["no_failure_filter"])
            self.assertEqual(json.loads(rows[0]["decision_trace_json"])["explanation"], "unit-test")
            self.assertAlmostEqual(rows[0]["historical_match_score"], 0.62)


if __name__ == "__main__":
    unittest.main()
