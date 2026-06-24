import unittest

from langlang_trader.distill_v1 import TradeLabel, TradeLabeler


class TradeLabelerTest(unittest.TestCase):
    def test_labels_big_wins_big_losses_right_tail_and_failure_shapes(self):
        trades = [
            {"trade_id": "1", "pnl_usdt": 1000, "return_rate": 0.80, "hold_minutes": 600, "side": "long"},
            {"trade_id": "2", "pnl_usdt": 450, "return_rate": 0.30, "hold_minutes": 180, "side": "long"},
            {"trade_id": "3", "pnl_usdt": -700, "return_rate": -0.45, "hold_minutes": 12, "side": "long"},
            {"trade_id": "4", "pnl_usdt": -250, "return_rate": -0.18, "hold_minutes": 20, "side": "short"},
            {"trade_id": "5", "pnl_usdt": 20, "return_rate": 0.02, "hold_minutes": 60, "side": "long"},
        ]

        labels = TradeLabeler().label_trades(trades)

        self.assertIn(TradeLabel.BIG_WIN.value, labels["1"])
        self.assertIn(TradeLabel.RIGHT_TAIL.value, labels["1"])
        self.assertIn(TradeLabel.BIG_LOSS.value, labels["3"])
        self.assertIn(TradeLabel.FAST_FAILURE.value, labels["3"])
        self.assertIn(TradeLabel.CHASE_FAILURE.value, labels["3"])
        self.assertIn(TradeLabel.ORDINARY.value, labels["5"])


if __name__ == "__main__":
    unittest.main()
