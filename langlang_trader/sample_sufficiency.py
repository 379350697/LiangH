from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LangLang sample sufficiency and attribution reports")
    parser.add_argument(
        "--ledger",
        action="append",
        required=True,
        help="Ledger spec as fleet_id:run_id:path. Can be passed multiple times.",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    ledgers = [_parse_ledger_spec(spec) for spec in args.ledger]
    paths = write_sample_sufficiency_reports(ledgers=ledgers, out_dir=args.out_dir)
    print(json.dumps(paths, ensure_ascii=False, sort_keys=True))
    return 0


def summarize_sample_sufficiency(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    fleet_summaries: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
    all_orders: list[dict[str, Any]] = []
    all_fills: list[dict[str, Any]] = []
    all_risk_events: list[dict[str, Any]] = []
    all_trade_events: list[dict[str, Any]] = []
    for spec in ledgers:
        fleet_id = str(spec["fleet_id"])
        run_id = str(spec["run_id"])
        ledger_path = str(spec["ledger_path"])
        fleet_data = _read_fleet(fleet_id=fleet_id, run_id=run_id, ledger_path=ledger_path)
        fleet_summaries.append(fleet_data["summary"])
        all_trades.extend(fleet_data["trades"])
        all_signals.extend(fleet_data["signals"])
        all_orders.extend(fleet_data["orders"])
        all_fills.extend(fleet_data["fills"])
        all_risk_events.extend(fleet_data["risk_events"])
        all_trade_events.extend(fleet_data["trade_events"])

    variant_sample_sufficiency = _variant_sample_sufficiency(all_trades)
    variant_attribution_report = _variant_attribution(
        trades=all_trades,
        signals=all_signals,
        orders=all_orders,
        fills=all_fills,
    )
    long_filter_diagnosis = _long_filter_diagnosis(all_risk_events)
    exit_management_attribution = _exit_management_attribution(all_trades, all_orders, all_trade_events)
    return {
        "generated_at": generated_at,
        "fleets": fleet_summaries,
        "variant_sample_sufficiency": variant_sample_sufficiency,
        "variant_attribution_report": variant_attribution_report,
        "long_filter_diagnosis": long_filter_diagnosis,
        "exit_management_attribution": exit_management_attribution,
    }


def write_sample_sufficiency_reports(
    *,
    ledgers: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, str]:
    summary = summarize_sample_sufficiency(ledgers)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {
        "variant_sample_sufficiency.json": summary["variant_sample_sufficiency"],
        "variant_attribution_report.json": summary["variant_attribution_report"],
        "long_filter_diagnosis.json": summary["long_filter_diagnosis"],
        "exit_management_attribution.json": summary["exit_management_attribution"],
        "sample_sufficiency_summary.md": _render_markdown(summary),
    }
    paths: dict[str, str] = {}
    for filename, payload in payloads.items():
        path = out / filename
        if filename.endswith(".json"):
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        else:
            path.write_text(str(payload), encoding="utf-8")
        paths[filename] = str(path)
    return paths


def _parse_ledger_spec(spec: str) -> dict[str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--ledger must use fleet_id:run_id:path")
    return {"fleet_id": parts[0], "run_id": parts[1], "ledger_path": parts[2]}


def _read_fleet(*, fleet_id: str, run_id: str, ledger_path: str) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        return {
            "summary": {
                "fleet_id": fleet_id,
                "run_id": run_id,
                "ledger_path": ledger_path,
                "ledger_exists": False,
                "signals": 0,
                "entry_orders": 0,
                "exit_orders": 0,
                "closed_trades": 0,
                "open_trades": 0,
            },
            "trades": [],
            "signals": [],
            "orders": [],
            "fills": [],
            "risk_events": [],
            "trade_events": [],
        }
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        trades = _table_rows(conn, "trade_lifecycle", run_id)
        signals = _table_rows(conn, "signals", run_id)
        orders = _table_rows(conn, "orders", run_id)
        fills = _table_rows(conn, "fills", run_id)
        risk_events = _table_rows(conn, "risk_events", run_id)
        trade_events = _table_rows(conn, "trade_events", run_id)
    for rows in (trades, signals, orders, fills, risk_events, trade_events):
        for row in rows:
            row["fleet_id"] = fleet_id
            row["ledger_path"] = ledger_path
    closed = sum(1 for row in trades if row.get("status") == "closed")
    return {
        "summary": {
            "fleet_id": fleet_id,
            "run_id": run_id,
            "ledger_path": ledger_path,
            "ledger_exists": True,
            "signals": len(signals),
            "entry_orders": sum(1 for row in orders if int(row.get("reduce_only") or 0) == 0),
            "exit_orders": sum(1 for row in orders if int(row.get("reduce_only") or 0) == 1),
            "closed_trades": closed,
            "open_trades": len(trades) - closed,
        },
        "trades": trades,
        "signals": signals,
        "orders": orders,
        "fills": fills,
        "risk_events": risk_events,
        "trade_events": trade_events,
    }


def _table_rows(conn: sqlite3.Connection, table: str, run_id: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    return [dict(row) for row in conn.execute(f"select * from {table} where run_id = ?", (run_id,)).fetchall()]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("select name from sqlite_master where type = 'table' and name = ?", (table,)).fetchone()
    return row is not None


def _variant_sample_sufficiency(trades: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[(str(trade.get("fleet_id") or ""), str(trade.get("variant_id") or ""))].append(trade)
    clusters = _correlated_clusters([trade for trade in trades if trade.get("status") == "closed"])
    independent_by_variant: Counter[tuple[str, str]] = Counter()
    for cluster in clusters:
        for variant_id in cluster["variant_ids"]:
            independent_by_variant[(cluster["fleet_id"], variant_id)] += 1

    rows = []
    for (fleet_id, variant_id), variant_trades in sorted(grouped.items()):
        closed = [trade for trade in variant_trades if trade.get("status") == "closed"]
        symbols = {str(trade.get("symbol") or "") for trade in closed}
        sides = {str(trade.get("side") or "") for trade in closed}
        independent_groups = int(independent_by_variant.get((fleet_id, variant_id), 0))
        trace = _first_decision_trace(variant_trades)
        status, action = _sample_status(
            closed_trades=len(closed),
            independent_groups=independent_groups,
            symbol_count=len(symbols),
            side_count=len(sides),
        )
        rows.append(
            {
                "fleet_id": fleet_id,
                "variant_id": variant_id,
                "total_trades": len(variant_trades),
                "closed_trades": len(closed),
                "open_trades": len(variant_trades) - len(closed),
                "distinct_closed_symbols": len(symbols),
                "distinct_closed_sides": len(sides),
                "independent_closed_trade_groups": independent_groups,
                "sample_status": status,
                "variant_action": action,
                "experiment_family": str(trace.get("experiment_family") or ""),
                "entry_family": str(trace.get("entry_family") or ""),
                "strategy_tree_variant_id": str(trace.get("strategy_tree_variant_id") or ""),
            }
        )
    return {
        "thresholds": {
            "diagnostic_only_lt_closed": 20,
            "early_elimination_min_closed": 30,
            "expansion_min_closed": 100,
            "expansion_min_symbols": 10,
            "expansion_min_sides": 2,
            "expansion_min_independent_groups": 50,
        },
        "variants": rows,
        "correlated_trade_clusters": clusters,
        "can_generate_new_variants": any(row["variant_action"] == "expand_or_mutate_allowed" for row in rows),
    }


def _sample_status(
    *,
    closed_trades: int,
    independent_groups: int,
    symbol_count: int,
    side_count: int,
) -> tuple[str, str]:
    if closed_trades < 20:
        return "diagnostic_only", "do_not_expand"
    if closed_trades < 30:
        return "observe_more", "do_not_expand"
    if closed_trades <= 50:
        return "early_elimination_reference", "early_elimination_only"
    if closed_trades < 100:
        return "calibration_watch", "do_not_expand"
    if independent_groups >= 50 and symbol_count >= 10 and side_count >= 2:
        return "expansion_eligible", "expand_or_mutate_allowed"
    return "sample_count_ok_diversity_insufficient", "do_not_expand"


def _correlated_clusters(closed_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in closed_trades:
        key = (
            str(trade.get("fleet_id") or ""),
            str(trade.get("symbol") or ""),
            str(trade.get("side") or ""),
            _minute_key(trade.get("opened_at")),
            _minute_key(trade.get("closed_at")),
        )
        grouped[key].append(trade)
    clusters: list[dict[str, Any]] = []
    for (fleet_id, symbol, side, opened_minute, closed_minute), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        clusters.append(
            {
                "fleet_id": fleet_id,
                "symbol": symbol,
                "side": side,
                "opened_minute": opened_minute,
                "closed_minute": closed_minute,
                "trade_count": len(rows),
                "independent_group_count": 1,
                "variant_ids": sorted({str(row.get("variant_id") or "") for row in rows}),
                "bot_ids": sorted({str(row.get("bot_id") or "") for row in rows}),
            }
        )
    return clusters


def _variant_attribution(
    *,
    trades: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    fills: list[dict[str, Any]],
) -> dict[str, Any]:
    variant_rows = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[
            (
                str(trade.get("fleet_id") or ""),
                str(trade.get("bot_id") or ""),
                str(trade.get("variant_id") or ""),
            )
        ].append(trade)
    signal_counts = _count_by_keys(signals, ("fleet_id", "bot_id", "variant_id"))
    entry_order_counts = Counter(
        _key(row, ("fleet_id", "bot_id", "variant_id")) for row in orders if int(row.get("reduce_only") or 0) == 0
    )
    exit_order_counts = Counter(
        _key(row, ("fleet_id", "bot_id", "variant_id")) for row in orders if int(row.get("reduce_only") or 0) == 1
    )
    fee_sums: Counter[tuple[str, str, str]] = Counter()
    for fill in fills:
        fee_sums[_key(fill, ("fleet_id", "bot_id", "variant_id"))] += _float(fill.get("fee"))

    side_buckets: Counter[str] = Counter()
    symbol_buckets: Counter[str] = Counter()
    entry_position_buckets: Counter[str] = Counter()
    experiment_family_buckets: Counter[str] = Counter()
    entry_family_buckets: Counter[str] = Counter()
    strategy_tree_variant_buckets: Counter[str] = Counter()
    entry_reason_buckets: Counter[str] = Counter()
    exit_reason_buckets: Counter[str] = Counter()
    strong_pattern_buckets: Counter[str] = Counter()
    risk_pattern_buckets: Counter[str] = Counter()
    wyckoff_phase_buckets: Counter[str] = Counter()
    wyckoff_setup_buckets: Counter[str] = Counter()
    data_quality_flags: Counter[str] = Counter()
    score_buckets: dict[str, Counter[str]] = {
        "strong_pattern_score": Counter(),
        "risk_pattern_score": Counter(),
        "wyckoff_long_score": Counter(),
        "wyckoff_short_score": Counter(),
        "wyckoff_risk_score": Counter(),
        "wyckoff_exit_score": Counter(),
    }

    for trade in trades:
        side_buckets[str(trade.get("side") or "unknown")] += 1
        symbol_buckets[str(trade.get("symbol") or "unknown")] += 1
        entry_trace = _json_dict(trade.get("entry_decision_trace_json"))
        if entry_trace.get("entry_position_id"):
            entry_position_buckets[str(entry_trace["entry_position_id"])] += 1
        if entry_trace.get("experiment_family"):
            experiment_family_buckets[str(entry_trace["experiment_family"])] += 1
        if entry_trace.get("entry_family"):
            entry_family_buckets[str(entry_trace["entry_family"])] += 1
        if entry_trace.get("strategy_tree_variant_id"):
            strategy_tree_variant_buckets[str(entry_trace["strategy_tree_variant_id"])] += 1
        for code in _json_list(trade.get("entry_reason_codes_json")):
            entry_reason_buckets[code] += 1
        for code in _json_list(trade.get("exit_reason_codes_json")):
            exit_reason_buckets[code] += 1
        for flag in _json_list(trade.get("data_quality_flags_json")):
            data_quality_flags[flag] += 1
        features = _json_dict(trade.get("entry_feature_snapshot_json"))
        _count_tag(str(features.get("strong_pattern_tag") or ""), strong_pattern_buckets)
        _count_tag(str(features.get("risk_pattern_tag") or ""), risk_pattern_buckets)
        _count_tag(str(features.get("wyckoff_phase_tag") or ""), wyckoff_phase_buckets)
        for key in ("wyckoff_long_setup_tag", "wyckoff_short_setup_tag", "wyckoff_exit_tag"):
            _count_tag(str(features.get(key) or ""), wyckoff_setup_buckets)
        for key, bucket in score_buckets.items():
            if key in features:
                bucket[_score_bucket(features.get(key))] += 1

    for key, rows in sorted(grouped.items()):
        closed = [row for row in rows if row.get("status") == "closed"]
        realized = [_float(row.get("realized_pnl_usdt")) for row in closed if row.get("realized_pnl_usdt") is not None]
        winners = sum(1 for value in realized if value > 0)
        r_values = [_float(row.get("r_multiple")) for row in closed if row.get("r_multiple") is not None]
        mae_values = [_float(row.get("mae_usdt")) for row in rows if row.get("mae_usdt") is not None]
        mfe_values = [_float(row.get("mfe_usdt")) for row in rows if row.get("mfe_usdt") is not None]
        variant_rows.append(
            {
                "fleet_id": key[0],
                "bot_id": key[1],
                "variant_id": key[2],
                "signals": int(signal_counts.get(key, 0)),
                "entries": int(entry_order_counts.get(key, 0)),
                "closed": len(closed),
                "open": len(rows) - len(closed),
                "win_rate": round(winners / len(closed), 6) if closed else None,
                "avg_pnl_usdt": round(sum(realized) / len(realized), 6) if realized else None,
                "sum_pnl_usdt": round(sum(realized), 6),
                "avg_r_multiple": round(sum(r_values) / len(r_values), 6) if r_values else None,
                "avg_mae_usdt": round(sum(mae_values) / len(mae_values), 6) if mae_values else None,
                "avg_mfe_usdt": round(sum(mfe_values) / len(mfe_values), 6) if mfe_values else None,
                "fee_drag_usdt": round(float(fee_sums.get(key, 0.0)), 6),
                "exit_orders": int(exit_order_counts.get(key, 0)),
            }
        )
    return {
        "variants": variant_rows,
        "side_buckets": dict(side_buckets),
        "symbol_buckets": dict(symbol_buckets),
        "entry_position_buckets": dict(entry_position_buckets),
        "experiment_family_buckets": dict(experiment_family_buckets),
        "entry_family_buckets": dict(entry_family_buckets),
        "strategy_tree_variant_buckets": dict(strategy_tree_variant_buckets),
        "entry_reason_buckets": dict(entry_reason_buckets),
        "exit_reason_buckets": dict(exit_reason_buckets),
        "strong_pattern_buckets": dict(strong_pattern_buckets),
        "risk_pattern_buckets": dict(risk_pattern_buckets),
        "wyckoff_phase_buckets": dict(wyckoff_phase_buckets),
        "wyckoff_setup_buckets": dict(wyckoff_setup_buckets),
        "score_buckets": {key: dict(value) for key, value in score_buckets.items()},
        "data_quality_flags": dict(data_quality_flags),
    }


def _long_filter_diagnosis(risk_events: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    filter_code_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()
    data_quality_flags: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    selection_long_candidates = 0
    selection_long_selected = 0
    selection_long_filtered = 0
    for event in risk_events:
        payload = _json_dict(event.get("payload_json"))
        if event.get("reason") == "symbol_selection":
            for candidate in _iter_ranked_long_candidates(payload):
                selection_long_candidates += 1
                if bool(candidate.get("selected")):
                    selection_long_selected += 1
                if candidate.get("filter_codes"):
                    selection_long_filtered += 1
                for code in _coerce_list(candidate.get("filter_codes")):
                    filter_code_counts[code] += 1
                for code in _coerce_list(candidate.get("reason_codes")):
                    selection_reason_counts[code] += 1
        if not _payload_is_long_intent(payload):
            continue
        reason = str(
            payload.get("risk_rejection_reason")
            or payload.get("skip_reason")
            or payload.get("reason")
            or event.get("reason")
            or "unknown"
        )
        reason_counts[reason] += 1
        by_variant[str(event.get("variant_id") or "")] += 1
        for key in ("filter_codes", "reason_codes", "selection_reason_codes"):
            for code in _coerce_list(payload.get(key)):
                filter_code_counts[code] += 1
        for flag in _coerce_list(payload.get("data_quality_flags")):
            data_quality_flags[flag] += 1
    return {
        "selection_long_candidates": selection_long_candidates,
        "selection_long_selected": selection_long_selected,
        "selection_long_filtered": selection_long_filtered,
        "reason_counts": dict(reason_counts),
        "filter_code_counts": dict(filter_code_counts),
        "selection_reason_counts": dict(selection_reason_counts),
        "data_quality_flags": dict(data_quality_flags),
        "by_variant": dict(by_variant),
    }


def _iter_ranked_long_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ranked_long = payload.get("ranked_long")
    if isinstance(ranked_long, list):
        candidates.extend(item for item in ranked_long if isinstance(item, dict))
    profiles = payload.get("profiles")
    if isinstance(profiles, dict):
        for profile_payload in profiles.values():
            if not isinstance(profile_payload, dict):
                continue
            ranked = profile_payload.get("ranked_long")
            if isinstance(ranked, list):
                candidates.extend(item for item in ranked if isinstance(item, dict))
    return candidates


def _exit_management_attribution(
    trades: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    trade_events: list[dict[str, Any]],
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter(str(event.get("event_type") or "unknown") for event in trade_events)
    exit_reason_counts: Counter[str] = Counter()
    reduce_only_orders: Counter[str] = Counter()
    for order in orders:
        if int(order.get("reduce_only") or 0) != 1:
            continue
        reason = str(order.get("exit_reason") or "unknown")
        reduce_only_orders[reason] += 1
    closed = [trade for trade in trades if trade.get("status") == "closed"]
    for trade in closed:
        for code in _json_list(trade.get("exit_reason_codes_json")):
            exit_reason_counts[code] += 1
    mfe_capture = [_float(trade.get("mfe_capture_ratio")) for trade in closed if trade.get("mfe_capture_ratio") is not None]
    r_values = [_float(trade.get("r_multiple")) for trade in closed if trade.get("r_multiple") is not None]
    return {
        "event_counts": dict(event_counts),
        "exit_reason_counts": dict(exit_reason_counts),
        "reduce_only_exit_order_counts": dict(reduce_only_orders),
        "closed_trades": len(closed),
        "avg_mfe_capture_ratio": round(sum(mfe_capture) / len(mfe_capture), 6) if mfe_capture else None,
        "avg_r_multiple": round(sum(r_values) / len(r_values), 6) if r_values else None,
    }


def _count_by_keys(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> Counter[tuple[str, ...]]:
    return Counter(_key(row, keys) for row in rows)


def _key(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(key) or "") for key in keys)


def _first_decision_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        trace = _json_dict(row.get("entry_decision_trace_json"))
        if trace:
            return trace
    return {}


def _minute_key(value: Any) -> str:
    text = str(value or "")
    if len(text) >= 16:
        return text[:16]
    return text


def _json_list(payload: Any) -> list[str]:
    if payload is None or payload == "":
        return []
    try:
        value = json.loads(str(payload))
    except json.JSONDecodeError:
        return ["invalid_json"]
    return _coerce_list(value)


def _json_dict(payload: Any) -> dict[str, Any]:
    if payload is None or payload == "":
        return {}
    try:
        value = json.loads(str(payload))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _payload_is_long_intent(payload: dict[str, Any]) -> bool:
    side_values = [
        payload.get("requested_side"),
        payload.get("selection_bias"),
        payload.get("side"),
        payload.get("signal_side"),
        payload.get("allowed_side"),
    ]
    return any(str(value).lower() == "long" for value in side_values if value is not None)


def _count_tag(tag: str, counter: Counter[str]) -> None:
    if tag and tag != "none":
        counter[tag] += 1


def _score_bucket(value: Any) -> str:
    score = _float(value)
    if score >= 0.70:
        return "gte_0_70"
    if score >= 0.65:
        return "0_65_to_0_70"
    if score > 0:
        return "0_to_0_65"
    return "zero_or_missing"


def _float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _render_markdown(summary: dict[str, Any]) -> str:
    sufficiency = summary["variant_sample_sufficiency"]
    attribution = summary["variant_attribution_report"]
    long_filters = summary["long_filter_diagnosis"]
    exits = summary["exit_management_attribution"]
    lines = [
        "# LangLang Sample Sufficiency Report",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- can_generate_new_variants: `{sufficiency['can_generate_new_variants']}`",
        "",
        "## Fleets",
        "",
    ]
    for fleet in summary["fleets"]:
        lines.append(
            "- {fleet_id}: signals={signals}, entries={entry_orders}, exits={exit_orders}, "
            "closed={closed_trades}, open={open_trades}".format(**fleet)
        )
    lines.extend(["", "## Variant Sufficiency", ""])
    for row in sufficiency["variants"]:
        lines.append(
            "- {fleet_id}/{variant_id}: closed={closed_trades}, independent_groups={independent_closed_trade_groups}, "
            "symbols={distinct_closed_symbols}, sides={distinct_closed_sides}, status={sample_status}, action={variant_action}".format(
                **row
            )
        )
    lines.extend(["", "## Attribution", ""])
    lines.append(f"- side_buckets: `{json.dumps(attribution['side_buckets'], ensure_ascii=False, sort_keys=True)}`")
    lines.append(
        "- experiment_family_buckets: "
        f"`{json.dumps(attribution['experiment_family_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        "- entry_family_buckets: "
        f"`{json.dumps(attribution['entry_family_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        "- strategy_tree_variant_buckets: "
        f"`{json.dumps(attribution['strategy_tree_variant_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        f"- entry_reason_buckets: `{json.dumps(attribution['entry_reason_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        f"- exit_reason_buckets: `{json.dumps(attribution['exit_reason_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        f"- strong_pattern_buckets: `{json.dumps(attribution['strong_pattern_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        f"- wyckoff_setup_buckets: `{json.dumps(attribution['wyckoff_setup_buckets'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.extend(["", "## Long Filter Diagnosis", ""])
    lines.append(f"- reason_counts: `{json.dumps(long_filters['reason_counts'], ensure_ascii=False, sort_keys=True)}`")
    lines.append(
        f"- filter_code_counts: `{json.dumps(long_filters['filter_code_counts'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.extend(["", "## Exit Management", ""])
    lines.append(f"- event_counts: `{json.dumps(exits['event_counts'], ensure_ascii=False, sort_keys=True)}`")
    lines.append(
        f"- reduce_only_exit_order_counts: `{json.dumps(exits['reduce_only_exit_order_counts'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
