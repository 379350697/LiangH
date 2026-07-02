from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from langlang_trader.factor_research import (
    FactorDefinition,
    FactorResearchStore,
    FactorResearchSample,
    MakerFillInput,
    MarketObservation,
    audit_lookahead,
    batch7_factor_registry,
    build_forward_labels,
    decompose_maker_pnl,
    main,
    purged_walk_forward_splits,
    score_factors,
    write_shadow_candidate_config,
)
from langlang_trader.hft_scalping import HftScalpPaperRunner, HftScalpVariant, RulesQueueImbalanceOneTickStrategy
from langlang_trader.ledger import Ledger
from liangh_trader.market_maker.models import BookTick


def _sample(
    sample_id: str,
    *,
    decision_time_ns: int,
    features: dict[str, float],
    feature_times_ns: dict[str, int] | None = None,
    side: str = "long",
    fired: bool = False,
) -> FactorResearchSample:
    return FactorResearchSample(
        sample_id=sample_id,
        run_id="batch7-test",
        bot_id="batch7_hft_queue_imbalance_btc_paper",
        strategy_tree_id="hft_queue_imbalance_btc_v1",
        symbol="BTCUSDT",
        venue="binance_usdm",
        event_seq=decision_time_ns,
        exchange_event_time_ms=1_700_000_000_000 + decision_time_ns // 1_000_000,
        receive_time_ns=decision_time_ns - 5,
        decision_time_ns=decision_time_ns,
        sample_type="book_decision",
        fired=fired,
        side=side,
        mid_price=100.0,
        features=features,
        feature_times_ns=feature_times_ns or {key: decision_time_ns - 5 for key in features},
    )


class Batch7FactorResearchTest(unittest.TestCase):
    def test_batch7_registry_covers_signal_and_maker_factor_families(self):
        registry = batch7_factor_registry()
        factor_ids = {factor.factor_id for factor in registry}

        self.assertIn("hft.queue_imbalance", factor_ids)
        self.assertIn("hft.spread_bps", factor_ids)
        self.assertIn("hft.sweep_notional_usdt", factor_ids)
        self.assertIn("hft.replenishment_ratio", factor_ids)
        self.assertIn("hft.lead_move_bps", factor_ids)
        self.assertIn("hft.lag_divergence_bps", factor_ids)
        self.assertIn("maker.ofi", factor_ids)
        self.assertIn("maker.inventory_base_qty", factor_ids)
        self.assertIn("maker.order_ttl_ms", factor_ids)
        self.assertIn("maker.quote_edge_bps", factor_ids)
        self.assertIn("cost.round_trip_fee_bps", factor_ids)
        self.assertTrue(all(factor.version_hash for factor in registry))
        self.assertTrue(all(factor.strategy_tree_path for factor in registry))

    def test_research_store_records_fired_and_non_fired_point_in_time_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            store.register_factors(batch7_factor_registry())
            store.record_observation(
                MarketObservation(
                    symbol="BTCUSDT",
                    venue="binance_usdm",
                    event_time_ms=1_700_000_000_000,
                    receive_time_ns=95,
                    bid=99.99,
                    ask=100.01,
                    bid_qty=7.0,
                    ask_qty=3.0,
                )
            )
            store.record_sample(_sample("s1", decision_time_ns=100, features={"hft.queue_imbalance": 0.4}, fired=False))
            store.record_sample(_sample("s2", decision_time_ns=200, features={"hft.queue_imbalance": 0.8}, fired=True))

            rows = store.list_samples()

            self.assertEqual([row.sample_id for row in rows], ["s1", "s2"])
            self.assertEqual([row.fired for row in rows], [False, True])
            self.assertEqual(rows[0].receive_time_ns, 95)
            self.assertEqual(rows[0].decision_time_ns, 100)

    def test_hft_runner_records_non_fired_and_fired_factor_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            variant = HftScalpVariant(
                variant_id="hft_queue_imbalance_btc_v1",
                symbol="BTC-USDT-SWAP",
                exchange_symbol="BTCUSDT",
                strategy_kind="queue_imbalance_one_tick",
                min_queue_imbalance=0.60,
            )
            runner = HftScalpPaperRunner(
                run_id="batch7-test",
                ledger=Ledger(os.path.join(tmp, "hft.sqlite3")),
                bots=[("batch7_hft_queue_imbalance_btc_paper", variant, RulesQueueImbalanceOneTickStrategy.version)],
                factor_research_store=store,
            )

            runner.on_book(
                BookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_700_000_000_000,
                    receive_time_ns=1_000,
                    best_bid=99.99,
                    best_bid_qty=10.0,
                    best_ask=100.01,
                    best_ask_qty=10.0,
                    update_id=1,
                )
            )
            runner.on_book(
                BookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_700_000_000_001,
                    receive_time_ns=2_000,
                    best_bid=99.99,
                    best_bid_qty=80.0,
                    best_ask=100.01,
                    best_ask_qty=10.0,
                    update_id=2,
                )
            )

            samples = store.list_samples()

            self.assertEqual([sample.fired for sample in samples], [False, True])
            self.assertEqual(samples[0].strategy_tree_id, "hft_queue_imbalance_btc_v1")
            self.assertAlmostEqual(samples[0].features["hft.queue_imbalance"], 0.0)
            self.assertGreater(samples[1].features["hft.queue_imbalance"], 0.6)

    def test_lookahead_audit_flags_future_feature_timestamp(self):
        registry = [
            FactorDefinition(
                factor_id="future.alpha",
                family="test",
                strategy_tree_path=("scalping", "batch7_hft_scalp", "test"),
                input_fields=("future_alpha",),
                max_lag_ns=0,
                usable_at_decision=True,
                version="unit",
            )
        ]
        samples = [
            _sample(
                "s1",
                decision_time_ns=100,
                features={"future.alpha": 1.0},
                feature_times_ns={"future.alpha": 101},
            )
        ]

        audit = audit_lookahead(samples, registry)

        self.assertEqual(audit["future.alpha"].decision, "leak_suspect")
        self.assertIn("feature_time_after_decision", audit["future.alpha"].flags)

    def test_labeler_waits_until_horizon_and_computes_net_taker_pnl(self):
        sample = _sample("s1", decision_time_ns=1_000, features={"hft.queue_imbalance": 0.9}, side="long")
        too_short = [
            MarketObservation("BTCUSDT", "binance_usdm", 1, 1_050, bid=100.0, ask=100.02),
        ]
        enough = [
            MarketObservation("BTCUSDT", "binance_usdm", 1, 1_000, bid=99.99, ask=100.01),
            MarketObservation("BTCUSDT", "binance_usdm", 2, 1_260, bid=100.19, ask=100.21),
        ]

        self.assertEqual(build_forward_labels([sample], too_short, horizons_ns=[250], round_trip_fee_bps=8.0), [])
        labels = build_forward_labels([sample], enough, horizons_ns=[250], round_trip_fee_bps=8.0)

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].label_start_ns, 1_000)
        self.assertEqual(labels[0].label_end_ns, 1_250)
        self.assertAlmostEqual(labels[0].forward_return_bps, 20.0, places=6)
        self.assertAlmostEqual(labels[0].net_taker_pnl_bps, 12.0, places=6)

    def test_purged_walk_forward_split_excludes_overlapping_label_windows(self):
        samples = [_sample(f"s{i}", decision_time_ns=i * 100, features={"hft.queue_imbalance": float(i)}) for i in range(8)]
        observations = [
            MarketObservation("BTCUSDT", "binance_usdm", i, i * 100 + 160, bid=100.0 + i, ask=100.02 + i)
            for i in range(8)
        ]
        labels = build_forward_labels(samples, observations, horizons_ns=[150], round_trip_fee_bps=0.0)

        splits = purged_walk_forward_splits(samples, labels, n_splits=2, embargo_ns=25)

        self.assertTrue(splits)
        label_by_sample = {label.sample_id: label for label in labels}
        for split in splits:
            val_start = min(samples_by_id.decision_time_ns for samples_by_id in split.validation_samples)
            val_end = max(label_by_sample[s.sample_id].label_end_ns for s in split.validation_samples)
            for train_sample in split.train_samples:
                train_label = label_by_sample[train_sample.sample_id]
                self.assertTrue(
                    train_label.label_end_ns + 25 <= val_start
                    or train_sample.decision_time_ns >= val_end + 25
                )

    def test_score_factors_keeps_true_signal_drops_noise_and_blocks_future_factor(self):
        true_factor = FactorDefinition(
            factor_id="true.alpha",
            family="test",
            strategy_tree_path=("scalping", "batch7_hft_scalp", "test"),
            input_fields=("true",),
            max_lag_ns=0,
            usable_at_decision=True,
            version="unit",
        )
        noise_factor = FactorDefinition(
            factor_id="noise.alpha",
            family="test",
            strategy_tree_path=("scalping", "batch7_hft_scalp", "test"),
            input_fields=("noise",),
            max_lag_ns=0,
            usable_at_decision=True,
            version="unit",
        )
        future_factor = FactorDefinition(
            factor_id="future.alpha",
            family="test",
            strategy_tree_path=("scalping", "batch7_hft_scalp", "test"),
            input_fields=("future",),
            max_lag_ns=0,
            usable_at_decision=True,
            version="unit",
        )
        samples = []
        for i in range(40):
            target = -10.0 + i * 0.5
            samples.append(
                _sample(
                    f"s{i}",
                    decision_time_ns=1_000 + i * 100,
                    features={
                        "true.alpha": target,
                        "noise.alpha": 1.0 if i % 2 == 0 else -1.0,
                        "future.alpha": target,
                    },
                    feature_times_ns={
                        "true.alpha": 995 + i * 100,
                        "noise.alpha": 995 + i * 100,
                        "future.alpha": 1_001 + i * 100,
                    },
                )
            )
        labels = [
            build_forward_labels(
                [sample],
                [
                    MarketObservation("BTCUSDT", "binance_usdm", 1, sample.decision_time_ns, bid=99.99, ask=100.01),
                    MarketObservation(
                        "BTCUSDT",
                        "binance_usdm",
                        2,
                        sample.decision_time_ns + 250,
                        bid=100.0 + sample.features["true.alpha"] / 100.0 - 0.01,
                        ask=100.0 + sample.features["true.alpha"] / 100.0 + 0.01,
                    ),
                ],
                horizons_ns=[250],
                round_trip_fee_bps=0.0,
            )[0]
            for sample in samples
        ]
        audit = audit_lookahead(samples, [true_factor, noise_factor, future_factor])

        scores = score_factors(samples, labels, [true_factor, noise_factor, future_factor], audit_results=audit, min_samples=20)

        by_id = {score.factor_id: score for score in scores}
        self.assertIn(by_id["true.alpha"].decision, {"keep", "watch"})
        self.assertEqual(by_id["noise.alpha"].decision, "drop")
        self.assertEqual(by_id["future.alpha"].decision, "leak_suspect")

    def test_maker_pnl_decomposition_splits_price_fee_and_cycle_win_rate(self):
        fills = [
            MakerFillInput("f1", "o1", "BTCUSDT", "buy", 100.0, 1.0, 0.02, "maker", 1_000, 1_000),
            MakerFillInput("f2", "o2", "BTCUSDT", "sell", 100.10, 1.0, 0.02, "maker", 1_100, 1_100),
            MakerFillInput("f3", "o3", "BTCUSDT", "buy", 100.0, 1.0, 0.02, "maker", 1_200, 1_200),
            MakerFillInput("f4", "o4", "BTCUSDT", "sell", 99.90, 1.0, 0.02, "maker", 1_300, 1_300),
        ]

        result = decompose_maker_pnl(fills)

        self.assertAlmostEqual(result.price_pnl_usdt, 0.0, places=8)
        self.assertAlmostEqual(result.fees_usdt, 0.08, places=8)
        self.assertAlmostEqual(result.net_pnl_usdt, -0.08, places=8)
        self.assertEqual(result.completed_cycles, 2)
        self.assertEqual(result.wins, 1)
        self.assertEqual(result.losses, 1)
        self.assertAlmostEqual(result.win_rate, 0.5, places=8)

    def test_shadow_candidate_config_does_not_overwrite_active_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "active.json"
            active.write_text(json.dumps({"active": True, "threshold": 0.6}), encoding="utf-8")

            candidate = write_shadow_candidate_config(
                active_config_path=active,
                output_dir=Path(tmp),
                run_id="research-run-1",
                kept_factors=["true.alpha"],
                dropped_factors=["noise.alpha"],
            )

            self.assertEqual(json.loads(active.read_text(encoding="utf-8")), {"active": True, "threshold": 0.6})
            self.assertTrue(candidate.name.endswith(".candidate.json"))
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertTrue(payload["shadow_only"])
            self.assertEqual(payload["source_config"], str(active))
            self.assertEqual(payload["kept_factors"], ["true.alpha"])

    def test_cli_batch7_audit_and_report_return_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            research_db = os.path.join(tmp, "research.sqlite3")
            output = io.StringIO()

            with redirect_stdout(output):
                code = main(["batch7", "audit", "--research-db", research_db])

            self.assertEqual(code, 0)
            audit = json.loads(output.getvalue())
            self.assertEqual(audit["batch_id"], "scalp-suite-batch7-24bot-paper-v1")
            self.assertEqual(audit["total_bots"], 24)
            self.assertEqual(audit["critical_count"], 0)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["batch7", "report", "--latest", "--research-db", research_db])

            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "ok")
            self.assertIn("factor_count", report)


if __name__ == "__main__":
    unittest.main()
