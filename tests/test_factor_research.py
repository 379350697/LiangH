from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from langlang_trader.factor_research import (
    FactorExperimentConfig,
    FactorDefinition,
    FactorResearchStore,
    FactorResearchSample,
    ForwardLabel,
    MakerFillInput,
    MarketObservation,
    audit_lookahead,
    batch7_factor_registry,
    build_daily_research_summary,
    build_forward_labels,
    decompose_maker_pnl,
    generate_research_candidate_config,
    main,
    purged_walk_forward_splits,
    run_factor_experiment,
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
                    best_bid_qty=55.0,
                    best_ask=100.01,
                    best_ask_qty=45.0,
                    update_id=1,
                )
            )
            runner.on_book(
                BookTick(
                    symbol="BTCUSDT",
                    event_time_ms=1_700_000_000_000,
                    receive_time_ns=1_500,
                    best_bid=99.99,
                    best_bid_qty=45.0,
                    best_ask=100.01,
                    best_ask_qty=55.0,
                    update_id=2,
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
                    update_id=3,
                )
            )

            samples = store.list_samples()

            self.assertEqual([sample.fired for sample in samples], [False, False, True])
            self.assertEqual([sample.side for sample in samples], ["long", "short", "long"])
            self.assertEqual(samples[0].strategy_tree_id, "hft_queue_imbalance_btc_v1")
            self.assertAlmostEqual(samples[0].features["hft.queue_imbalance"], 0.1)
            self.assertLess(samples[1].features["hft.queue_imbalance"], 0.0)
            self.assertGreater(samples[2].features["hft.queue_imbalance"], 0.6)

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

    def test_labeler_skips_samples_without_candidate_side(self):
        sample = _sample("s1", decision_time_ns=1_000, features={"hft.queue_imbalance": 0.1}, side="")
        observations = [
            MarketObservation("BTCUSDT", "binance_usdm", 1, 1_000, bid=99.99, ask=100.01),
            MarketObservation("BTCUSDT", "binance_usdm", 2, 1_260, bid=100.19, ask=100.21),
        ]

        labels = build_forward_labels([sample], observations, horizons_ns=[250], round_trip_fee_bps=0.0)

        self.assertEqual(labels, [])

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

    def test_score_and_report_keep_forward_horizons_separate(self):
        registry = [
            FactorDefinition("true.alpha", "test", ("batch7", "unit"), ("true",), 0, True, "unit"),
        ]
        samples = []
        labels = []
        for i in range(40):
            value = float(i)
            sample = _sample(
                f"s{i}",
                decision_time_ns=1_000 + i * 1_000,
                features={"true.alpha": value},
            )
            samples.append(sample)
            labels.extend(
                [
                    ForwardLabel(
                        sample_id=sample.sample_id,
                        horizon_ns=250,
                        label_start_ns=sample.decision_time_ns,
                        label_end_ns=sample.decision_time_ns + 250,
                        forward_return_bps=-value,
                        net_taker_pnl_bps=-value,
                        mfe_bps=0.0,
                        mae_bps=-value,
                        hit_take_profit=False,
                        hit_stop_loss=False,
                    ),
                    ForwardLabel(
                        sample_id=sample.sample_id,
                        horizon_ns=1_000,
                        label_start_ns=sample.decision_time_ns,
                        label_end_ns=sample.decision_time_ns + 1_000,
                        forward_return_bps=value,
                        net_taker_pnl_bps=value,
                        mfe_bps=value,
                        mae_bps=0.0,
                        hit_take_profit=False,
                        hit_stop_loss=False,
                    ),
                ]
            )

        short_scores = score_factors(samples, labels, registry, min_samples=20, horizon_ns=250)
        long_scores = score_factors(samples, labels, registry, min_samples=20, horizon_ns=1_000)
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            for sample in samples:
                store.record_sample(sample)
            store.record_labels(labels)
            report = run_factor_experiment(
                store,
                registry,
                FactorExperimentConfig(run_id="multi-horizon", min_samples=20, n_splits=0),
            ).report

        self.assertLess(short_scores[0].ic, 0.0)
        self.assertGreater(long_scores[0].ic, 0.0)
        self.assertEqual(report["primary_horizon_ns"], 250)
        self.assertLess(report["horizon_reports"]["250"]["factor_conclusions"]["true.alpha"]["mean_net_taker_pnl_bps"], 0.0)
        self.assertGreater(report["horizon_reports"]["1000"]["factor_conclusions"]["true.alpha"]["mean_net_taker_pnl_bps"], 0.0)

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

    def test_experiment_runner_records_oos_artifact_and_factor_conclusions(self):
        registry = [
            FactorDefinition("true.alpha", "test", ("batch7", "unit"), ("true",), 0, True, "unit"),
            FactorDefinition("noise.alpha", "test", ("batch7", "unit"), ("noise",), 0, True, "unit"),
            FactorDefinition("future.alpha", "test", ("batch7", "unit"), ("future",), 0, True, "unit"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            store.register_factors(registry)
            for i in range(64):
                decision_time_ns = 1_000_000 + i * 100_000
                true_value = float(i % 16)
                signed_net = true_value - 4.0
                sample = _sample(
                    f"s{i}",
                    decision_time_ns=decision_time_ns,
                    features={
                        "true.alpha": true_value,
                        "noise.alpha": 1.0 if i % 2 else -1.0,
                        "future.alpha": signed_net,
                    },
                    feature_times_ns={
                        "true.alpha": decision_time_ns - 10,
                        "noise.alpha": decision_time_ns - 10,
                        "future.alpha": decision_time_ns + 1,
                    },
                )
                store.record_sample(sample)
                store.record_labels(
                    [
                        ForwardLabel(
                            sample_id=sample.sample_id,
                            horizon_ns=50_000,
                            label_start_ns=decision_time_ns,
                            label_end_ns=decision_time_ns + 50_000,
                            forward_return_bps=signed_net,
                            net_taker_pnl_bps=signed_net,
                            mfe_bps=max(0.0, signed_net),
                            mae_bps=min(0.0, signed_net),
                            hit_take_profit=signed_net >= 10.0,
                            hit_stop_loss=signed_net <= -2.5,
                        )
                    ]
                )

            result = run_factor_experiment(
                store,
                registry,
                FactorExperimentConfig(
                    run_id="exp-unit",
                    min_samples=20,
                    n_splits=3,
                    embargo_ns=10_000,
                    artifact_dir=os.path.join(tmp, "artifacts"),
                ),
            )

            self.assertEqual(result.run_id, "exp-unit")
            self.assertTrue(Path(result.artifact_path).exists())
            conclusions = result.report["factor_conclusions"]
            self.assertIn(conclusions["true.alpha"]["decision"], {"keep", "watch"})
            self.assertEqual(conclusions["noise.alpha"]["decision"], "drop")
            self.assertEqual(conclusions["future.alpha"]["decision"], "leak_suspect")
            self.assertGreater(result.report["walk_forward"]["split_count"], 0)
            latest = store.latest_experiment_summary()
            self.assertEqual(latest["run_id"], "exp-unit")

    def test_candidate_generation_uses_only_clean_oos_factors_and_preserves_active_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "active.json"
            active_payload = {
                "run_id": "active",
                "bot_matrix": {
                    "strategies": [
                        {
                            "strategy_key": "hft_queue_imbalance_one_tick",
                            "parameters": {"min_queue_imbalance": 0.6, "take_profit_bps": 10.0},
                        }
                    ]
                },
            }
            active.write_text(json.dumps(active_payload), encoding="utf-8")
            report = {
                "factor_conclusions": {
                    "hft.queue_imbalance": {
                        "decision": "keep",
                        "recommended_threshold": 0.72,
                        "oos_mean_net_bps": 3.5,
                    },
                    "hft.spread_bps": {"decision": "drop", "recommended_threshold": 1.0},
                    "future.alpha": {"decision": "leak_suspect", "recommended_threshold": 9.0},
                }
            }

            candidate = generate_research_candidate_config(
                active_config_path=active,
                output_dir=Path(tmp),
                run_id="exp-unit",
                report=report,
            )

            self.assertEqual(json.loads(active.read_text(encoding="utf-8")), active_payload)
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertTrue(payload["shadow_only"])
            self.assertTrue(payload["promotion_gate"]["requires_manual_approval"])
            self.assertEqual(payload["factor_threshold_candidates"], {"hft.queue_imbalance": 0.72})
            self.assertEqual(payload["blocked_factors"], ["future.alpha"])

    def test_daily_summary_combines_factor_status_and_maker_pnl_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            registry = [
                FactorDefinition("true.alpha", "test", ("batch7", "unit"), ("true",), 0, True, "unit"),
            ]
            store.register_factors(registry)
            sample = _sample("s1", decision_time_ns=1_000, features={"true.alpha": 1.0})
            store.record_sample(sample)
            store.record_labels(
                [
                    ForwardLabel(
                        sample_id="s1",
                        horizon_ns=250,
                        label_start_ns=1_000,
                        label_end_ns=1_250,
                        forward_return_bps=5.0,
                        net_taker_pnl_bps=3.0,
                        mfe_bps=5.0,
                        mae_bps=0.0,
                        hit_take_profit=False,
                        hit_stop_loss=False,
                    )
                ]
            )
            run_factor_experiment(
                store,
                registry,
                FactorExperimentConfig(run_id="exp-summary", min_samples=1, n_splits=0, artifact_dir=os.path.join(tmp, "artifacts")),
            )
            fills = [
                MakerFillInput("f1", "o1", "BTCUSDT", "buy", 100.0, 1.0, 0.02, "maker", 1_000, 1_000),
                MakerFillInput("f2", "o2", "BTCUSDT", "sell", 100.05, 1.0, 0.02, "maker", 1_100, 1_100),
            ]

            summary = build_daily_research_summary(store, registry, maker_fills=fills)

            self.assertEqual(summary["latest_experiment"]["run_id"], "exp-summary")
            self.assertEqual(summary["factor_status_counts"]["insufficient_sample"], 0)
            self.assertAlmostEqual(summary["maker_pnl"]["price_pnl_usdt"], 0.05, places=8)
            self.assertAlmostEqual(summary["maker_pnl"]["fees_usdt"], 0.04, places=8)
            self.assertAlmostEqual(summary["maker_pnl"]["net_pnl_usdt"], 0.01, places=8)

    def test_daily_summary_groups_maker_pnl_by_symbol_before_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FactorResearchStore(os.path.join(tmp, "research.sqlite3"))
            registry = [
                FactorDefinition("true.alpha", "test", ("batch7", "unit"), ("true",), 0, True, "unit"),
            ]
            fills = [
                MakerFillInput("f1", "o1", "BTCUSDT", "buy", 100.0, 1.0, 0.10, "maker", 1_000, 1_000),
                MakerFillInput("f2", "o2", "BTCUSDT", "sell", 101.0, 1.0, 0.10, "maker", 1_100, 1_100),
                MakerFillInput("f3", "o3", "ETHUSDT", "buy", 1_000.0, 1.0, 0.10, "maker", 1_200, 1_200),
                MakerFillInput("f4", "o4", "ETHUSDT", "sell", 990.0, 1.0, 0.10, "maker", 1_300, 1_300),
            ]

            summary = build_daily_research_summary(store, registry, maker_fills=fills)

            by_symbol = summary["maker_pnl_by_symbol"]
            self.assertAlmostEqual(by_symbol["BTCUSDT"]["price_pnl_usdt"], 1.0, places=8)
            self.assertAlmostEqual(by_symbol["ETHUSDT"]["price_pnl_usdt"], -10.0, places=8)
            self.assertAlmostEqual(summary["maker_pnl_total"]["price_pnl_usdt"], -9.0, places=8)
            self.assertAlmostEqual(summary["maker_pnl_total"]["fees_usdt"], 0.4, places=8)
            self.assertAlmostEqual(summary["maker_pnl_total"]["net_pnl_usdt"], -9.4, places=8)
            self.assertEqual(summary["maker_pnl_total"]["completed_cycles"], 2)
            self.assertEqual(summary["maker_pnl_total"]["wins"], 1)
            self.assertEqual(summary["maker_pnl_total"]["losses"], 1)
            self.assertAlmostEqual(summary["maker_pnl_total"]["win_rate"], 0.5, places=8)

    def test_cli_experiment_and_candidate_commands_emit_json_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            research_db = os.path.join(tmp, "research.sqlite3")
            store = FactorResearchStore(research_db)
            store.register_factors(batch7_factor_registry())
            for i in range(12):
                decision_time_ns = 1_000 + i * 100
                sample = _sample(
                    f"s{i}",
                    decision_time_ns=decision_time_ns,
                    features={"hft.queue_imbalance": float(i), "hft.spread_bps": float(12 - i)},
                )
                store.record_sample(sample)
                store.record_labels(
                    [
                        ForwardLabel(
                            sample_id=sample.sample_id,
                            horizon_ns=250,
                            label_start_ns=decision_time_ns,
                            label_end_ns=decision_time_ns + 250,
                            forward_return_bps=float(i),
                            net_taker_pnl_bps=float(i),
                            mfe_bps=float(i),
                            mae_bps=0.0,
                            hit_take_profit=False,
                            hit_stop_loss=False,
                        )
                    ]
                )
            active = Path(tmp) / "active.json"
            active.write_text(json.dumps({"bot_matrix": {"strategies": []}}), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "batch7",
                        "experiment",
                        "--research-db",
                        research_db,
                        "--run-id",
                        "cli-exp",
                        "--artifact-dir",
                        os.path.join(tmp, "artifacts"),
                        "--min-samples",
                        "5",
                    ]
                )

            self.assertEqual(code, 0)
            experiment = json.loads(output.getvalue())
            self.assertEqual(experiment["run_id"], "cli-exp")
            self.assertTrue(Path(experiment["artifact_path"]).exists())

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "batch7",
                        "candidate-generate",
                        "--research-db",
                        research_db,
                        "--active-config",
                        str(active),
                        "--output-dir",
                        tmp,
                        "--run-id",
                        "cli-exp",
                    ]
                )

            self.assertEqual(code, 0)
            candidate = json.loads(output.getvalue())
            self.assertTrue(candidate["shadow_only"])
            self.assertTrue(candidate["candidate_config"].endswith(".candidate.json"))

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

    def test_cli_build_dataset_honors_from_to_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            research_db = os.path.join(tmp, "research.sqlite3")
            store = FactorResearchStore(research_db)
            for index, decision_time_ns in enumerate((100, 200, 300), start=1):
                store.record_sample(
                    _sample(
                        f"s{index}",
                        decision_time_ns=decision_time_ns,
                        features={"hft.queue_imbalance": 0.2},
                        side="long",
                    )
                )
                store.record_observation(
                    MarketObservation("BTCUSDT", "binance_usdm", index, decision_time_ns, bid=99.99, ask=100.01)
                )
                store.record_observation(
                    MarketObservation("BTCUSDT", "binance_usdm", index, decision_time_ns + 50, bid=100.09, ask=100.11)
                )

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "batch7",
                        "build-dataset",
                        "--research-db",
                        research_db,
                        "--horizons",
                        "50",
                        "--round-trip-fee-bps",
                        "0",
                        "--from",
                        "150",
                        "--to",
                        "250",
                    ]
                )

            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["label_count"], 1)
            self.assertEqual([label.sample_id for label in store.list_labels()], ["s2"])


if __name__ == "__main__":
    unittest.main()
