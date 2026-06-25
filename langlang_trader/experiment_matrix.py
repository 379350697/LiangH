from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from langlang_trader.features import FeatureSnapshot
from langlang_trader.strategy import LangLangV1_3Variant


FEATURE_PROFILE_BASELINE_V1_3 = "baseline_v1_3"
FEATURE_PROFILE_STRONG_PATTERN_V1_3 = "strong_pattern_v1_3"
FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3 = "wyckoff_enhanced_v1_3"
V1_3_EXPERIMENT_PROFILES = (
    FEATURE_PROFILE_BASELINE_V1_3,
    FEATURE_PROFILE_STRONG_PATTERN_V1_3,
    FEATURE_PROFILE_WYCKOFF_ENHANCED_V1_3,
)

_PATTERN_SCORE_NAMES = {
    "leader_platform_start_score",
    "golden_pit_reclaim_score",
    "small_divergence_absorb_score",
    "second_wave_start_score",
    "spoon_bottom_confirmed_score",
    "five_wave_late_risk_score",
    "false_breakout_risk_score",
    "strong_pattern_score",
    "risk_pattern_score",
}
_PATTERN_TAG_NAMES = {"strong_pattern_tag", "risk_pattern_tag"}
_PATTERN_LIST_NAMES = {
    "pattern_reason_codes",
    "strong_pattern_reason_codes",
    "risk_pattern_reason_codes",
    "leader_platform_start_reason_codes",
    "golden_pit_reclaim_reason_codes",
    "small_divergence_absorb_reason_codes",
    "second_wave_start_reason_codes",
    "spoon_bottom_confirmed_reason_codes",
    "five_wave_late_risk_reason_codes",
    "false_breakout_risk_reason_codes",
}
_PATTERN_STRUCTURE_DEFAULTS = {
    "wave_push_count": 0,
    "small_divergence_count": 0,
    "pivot_quality_score": 0.0,
}
_WYCKOFF_TAG_NAMES = {
    "wyckoff_phase_tag",
    "wyckoff_long_setup_tag",
    "wyckoff_short_setup_tag",
    "wyckoff_exit_tag",
}
_WYCKOFF_LIST_SUFFIX = "_reason_codes"


@dataclass(frozen=True)
class ExperimentMatrixResult:
    profile_dirs: dict[str, str]
    summary_path: str
    attribution_path: str
    report_path: str


def apply_feature_profile(snapshot: FeatureSnapshot, profile: str) -> FeatureSnapshot:
    if profile not in V1_3_EXPERIMENT_PROFILES:
        raise ValueError(f"unsupported v1.3 experiment feature profile: {profile}")
    features = dict(snapshot.features)
    if profile == FEATURE_PROFILE_BASELINE_V1_3:
        _mask_pattern_features(features)
        _mask_wyckoff_features(features)
    elif profile == FEATURE_PROFILE_STRONG_PATTERN_V1_3:
        _mask_wyckoff_features(features)
    return FeatureSnapshot(
        symbol=snapshot.symbol,
        bar=snapshot.bar,
        last_ts=snapshot.last_ts,
        features=features,
        created_at=snapshot.created_at,
    )


def run_v1_3_experiment_matrix(
    *,
    trades_csv: str,
    kline_cache_dir: str,
    out_dir: str,
    variants: list[LangLangV1_3Variant] | None = None,
    top_n: int = 10,
    max_variants: int | None = None,
    min_validation_signals: int = 0,
    max_validation_signals: int = 300,
) -> ExperimentMatrixResult:
    from langlang_trader.optimize import HistoricalReplayOptimizer, OptimizerConfig

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    profile_payloads: dict[str, dict[str, Any]] = {}
    profile_dirs: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for profile in V1_3_EXPERIMENT_PROFILES:
        profile_out = root / profile
        result = HistoricalReplayOptimizer(
            OptimizerConfig(
                trades_csv=trades_csv,
                kline_cache_dir=kline_cache_dir,
                out_dir=str(profile_out),
                strategy_version="rules_langlang_v1_3",
                variants=variants,
                top_n=top_n,
                max_variants=max_variants,
                min_validation_signals=min_validation_signals,
                max_validation_signals=max_validation_signals,
                strategy_library_registry_path=None,
                strategy_library_db_path=None,
                feature_profile=profile,
                experiment_label=profile,
            )
        ).run()
        profile_dirs[profile] = str(profile_out)
        rows = [_summary_row(profile, row, result.leaderboard_path, result.report_path) for row in result.leaderboard]
        summary_rows.extend(rows)
        profile_payloads[profile] = {
            "leaderboard_path": result.leaderboard_path,
            "selected_config_path": result.selected_config_path,
            "optimizer_report_path": result.report_path,
            "variants_scored": len(result.leaderboard),
            "leaderboard": [_attribution_row(row) for row in result.leaderboard],
        }

    summary_path = root / "experiment_matrix_summary.csv"
    _write_csv(summary_path, summary_rows)
    attribution_path = root / "experiment_matrix_attribution.json"
    attribution_path.write_text(
        json.dumps({"profiles": profile_payloads}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = root / "experiment_matrix_report.md"
    report_path.write_text(_render_matrix_report(summary_rows, profile_payloads), encoding="utf-8")
    return ExperimentMatrixResult(
        profile_dirs=profile_dirs,
        summary_path=str(summary_path),
        attribution_path=str(attribution_path),
        report_path=str(report_path),
    )


def _mask_pattern_features(features: dict[str, Any]) -> None:
    for key in list(features.keys()):
        base = _strip_bar_prefix(key)
        if base in _PATTERN_SCORE_NAMES:
            features[key] = 0.0
        elif base in _PATTERN_TAG_NAMES:
            features[key] = ""
        elif base in _PATTERN_LIST_NAMES:
            features[key] = []
    for key, value in _PATTERN_STRUCTURE_DEFAULTS.items():
        features[key] = value


def _mask_wyckoff_features(features: dict[str, Any]) -> None:
    for key in list(features.keys()):
        base = _strip_bar_prefix(key)
        if not base.startswith("wyckoff_"):
            continue
        if base in _WYCKOFF_TAG_NAMES:
            features[key] = "none" if base == "wyckoff_phase_tag" else ""
        elif base.endswith(_WYCKOFF_LIST_SUFFIX):
            features[key] = []
        else:
            features[key] = 0.0


def _strip_bar_prefix(key: str) -> str:
    for prefix in ("h1_", "m15_", "m5_", "m1_"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _summary_row(profile: str, row: dict[str, Any], leaderboard_path: str, report_path: str) -> dict[str, Any]:
    diagnostics = row.get("experiment_diagnostics") or {}
    variant = row.get("variant")
    params = variant.to_dict() if hasattr(variant, "to_dict") else {}
    return {
        "experiment_label": profile,
        "variant_id": row.get("variant_id", ""),
        "rank": row.get("rank", ""),
        "eligible": row.get("eligible", ""),
        "score": _fmt(row.get("score")),
        "validation_signals": row.get("validation_signals", 0),
        "raw_validation_signals": row.get("raw_validation_signals", 0),
        "validation_net_pnl": _fmt(row.get("validation_net_pnl")),
        "validation_profit_factor": _fmt(row.get("validation_profit_factor")),
        "max_drawdown": _fmt(row.get("max_drawdown")),
        "big_win_recall": _fmt(row.get("big_win_recall")),
        "big_loss_overlap": _fmt(row.get("big_loss_overlap")),
        "right_tail_capture_score": _fmt(row.get("right_tail_capture_score")),
        "loss_suppression_score": _fmt(row.get("loss_suppression_score")),
        "ret_20d_min": _fmt(params.get("ret_20d_min")),
        "ret_60d_min": _fmt(params.get("ret_60d_min")),
        "pos_20d_min": _fmt(params.get("pos_20d_min")),
        "min_upside_space_pct": _fmt(params.get("min_upside_space_pct")),
        "min_historical_match_score": _fmt(params.get("min_historical_match_score")),
        "allowed_side": params.get("allowed_side", ""),
        "diagnostic_snapshots": diagnostics.get("snapshot_count", 0),
        "zero_signal_top_filters": _join_counts(diagnostics.get("skip_filter_counts", {})),
        "zero_signal_top_explanations": _join_counts(diagnostics.get("skip_explanation_counts", {})),
        "entry_position_counts": _join_counts(diagnostics.get("entry_position_counts", {})),
        "strong_pattern_tag_counts": _join_counts(diagnostics.get("strong_pattern_tag_counts", {})),
        "risk_pattern_tag_counts": _join_counts(diagnostics.get("risk_pattern_tag_counts", {})),
        "wyckoff_phase_counts": _join_counts(diagnostics.get("wyckoff_phase_counts", {})),
        "wyckoff_long_setup_counts": _join_counts(diagnostics.get("wyckoff_long_setup_counts", {})),
        "wyckoff_short_setup_counts": _join_counts(diagnostics.get("wyckoff_short_setup_counts", {})),
        "strong_pattern_score_bins": _join_counts(diagnostics.get("strong_pattern_score_bins", {})),
        "risk_pattern_score_bins": _join_counts(diagnostics.get("risk_pattern_score_bins", {})),
        "wyckoff_long_score_bins": _join_counts(diagnostics.get("wyckoff_long_score_bins", {})),
        "wyckoff_short_score_bins": _join_counts(diagnostics.get("wyckoff_short_score_bins", {})),
        "wyckoff_exit_score_bins": _join_counts(diagnostics.get("wyckoff_exit_score_bins", {})),
        "strong_pattern_released_signals": diagnostics.get("strong_pattern_released_signals", 0),
        "wyckoff_released_signals": diagnostics.get("wyckoff_released_signals", 0),
        "risk_filtered_skips": diagnostics.get("risk_filtered_skips", 0),
        "leaderboard_path": leaderboard_path,
        "optimizer_report_path": report_path,
    }


def _attribution_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in row.items() if key not in {"variant"}}
    variant = row.get("variant")
    if hasattr(variant, "to_dict"):
        payload["variant_params"] = variant.to_dict()
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_matrix_report(summary_rows: list[dict[str, Any]], profile_payloads: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# LangLang v1.3 Strong Pattern + Wyckoff Experiment Matrix",
        "",
        "## Summary",
    ]
    for profile in V1_3_EXPERIMENT_PROFILES:
        rows = [row for row in summary_rows if row["experiment_label"] == profile]
        best = rows[0] if rows else {}
        signal_rows = [row for row in rows if int(row.get("validation_signals") or 0) > 0]
        best_signal = signal_rows[0] if signal_rows else {}
        total_signals = sum(int(row.get("validation_signals") or 0) for row in rows)
        lines.append(
            f"- {profile}: variants={len(rows)} total_signals={total_signals} "
            f"best_variant={best.get('variant_id', '')} best_score={best.get('score', '')} "
            f"best_signal_variant={best_signal.get('variant_id', 'none')} "
            f"best_signal_count={best_signal.get('validation_signals', 0)}"
        )

    lines.extend(["", "## Zero Signal Diagnostics"])
    for profile in V1_3_EXPERIMENT_PROFILES:
        rows = [row for row in summary_rows if row["experiment_label"] == profile]
        top_filters = rows[0].get("zero_signal_top_filters", "") if rows else ""
        top_explanations = rows[0].get("zero_signal_top_explanations", "") if rows else ""
        lines.append(f"- {profile}: filters={top_filters or 'none'} explanations={top_explanations or 'none'}")

    lines.extend(["", "## Attribution Buckets"])
    for profile in V1_3_EXPERIMENT_PROFILES:
        rows = [row for row in summary_rows if row["experiment_label"] == profile]
        if not rows:
            continue
        best = rows[0]
        lines.append(
            f"- {profile}: strong_pattern_score={best.get('strong_pattern_score_bins', '') or 'none'}; "
            f"risk_pattern_score={best.get('risk_pattern_score_bins', '') or 'none'}; "
            f"wyckoff_long_score={best.get('wyckoff_long_score_bins', '') or 'none'}; "
            f"wyckoff_short_score={best.get('wyckoff_short_score_bins', '') or 'none'}; "
            f"wyckoff_exit_score={best.get('wyckoff_exit_score_bins', '') or 'none'}"
        )

    lines.extend(["", "## Files"])
    for profile, payload in profile_payloads.items():
        lines.append(f"- {profile}: {payload['leaderboard_path']}")
    return "\n".join(lines) + "\n"


def _join_counts(counts: dict[str, Any], *, limit: int = 8) -> str:
    if not counts:
        return ""
    items = sorted(((str(key), int(value)) for key, value in counts.items()), key=lambda item: (-item[1], item[0]))
    return "|".join(f"{key}:{value}" for key, value in items[:limit])


def _fmt(value: Any) -> str:
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LangLang v1.3 baseline/strong-pattern/Wyckoff experiment matrix")
    parser.add_argument("--trades", default="output/langlang_distill/standard_trades.csv")
    parser.add_argument("--kline-cache", default="output/langlang_distill/kline_cache")
    parser.add_argument("--out", default="output/fleet/langlang_v1_3_shape_wyckoff_matrix_latest")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--min-validation-signals", type=int, default=0)
    parser.add_argument("--max-validation-signals", type=int, default=300)
    args = parser.parse_args(argv)
    result = run_v1_3_experiment_matrix(
        trades_csv=args.trades,
        kline_cache_dir=args.kline_cache,
        out_dir=args.out,
        top_n=args.top_n,
        max_variants=args.max_variants,
        min_validation_signals=args.min_validation_signals,
        max_validation_signals=args.max_validation_signals,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
