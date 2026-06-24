from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_REGISTRY_PATH = "configs/strategy_library/langlang_strategy_tree.json"
DEFAULT_DB_PATH = "output/strategy_library/strategy_library.sqlite3"


class StrategyLibraryError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyFamilyNode:
    family_id: str
    name: str
    source_basis: list[str]


@dataclass(frozen=True)
class StrategyNode:
    strategy_id: str
    family_id: str
    strategy_version: str
    status: str
    hypothesis: str
    promotion_rules: list[str]


@dataclass(frozen=True)
class StrategyVariantNode:
    variant_id: str
    display_name: str
    lineage_group: str | None
    strategy_id: str
    parent_id: str | None
    strategy_version: str
    status: str
    hypothesis: str
    factor_set: dict[str, Any]
    changed_factors: list[str]
    core_logic: list[str]
    source_basis: list[str]
    risk_profile: dict[str, Any]
    iteration_notes: str
    promotion_rules: list[str]


@dataclass(frozen=True)
class IngestResult:
    inserted_runs: int
    db_path: str


class StrategyLibrary:
    def __init__(
        self,
        *,
        families: dict[str, StrategyFamilyNode],
        strategies: dict[str, StrategyNode],
        variants: dict[str, StrategyVariantNode],
    ):
        self.families = families
        self.strategies = strategies
        self.variants = variants

    @classmethod
    def load(cls, path: str | Path) -> "StrategyLibrary":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        families = {
            row["family_id"]: StrategyFamilyNode(
                family_id=row["family_id"],
                name=row.get("name", row["family_id"]),
                source_basis=list(row.get("source_basis", [])),
            )
            for row in raw.get("families", [])
        }
        _reject_duplicate_ids(raw.get("families", []), "family_id")

        _reject_duplicate_ids(raw.get("strategies", []), "strategy_id")
        strategies = {}
        for row in raw.get("strategies", []):
            _require(row, ["strategy_id", "family_id", "strategy_version", "status", "hypothesis", "promotion_rules"])
            if row["family_id"] not in families:
                raise StrategyLibraryError(f"strategy {row['strategy_id']} references missing family {row['family_id']}")
            strategies[row["strategy_id"]] = StrategyNode(
                strategy_id=row["strategy_id"],
                family_id=row["family_id"],
                strategy_version=row["strategy_version"],
                status=row["status"],
                hypothesis=row["hypothesis"],
                promotion_rules=list(row.get("promotion_rules", [])),
            )

        _reject_duplicate_ids(raw.get("variants", []), "variant_id")
        variants = {}
        for row in raw.get("variants", []):
            _require(
                row,
                [
                    "variant_id",
                    "strategy_id",
                    "parent_id",
                    "strategy_version",
                    "status",
                    "hypothesis",
                    "factor_set",
                    "changed_factors",
                    "source_basis",
                    "risk_profile",
                    "promotion_rules",
                ],
            )
            if row["strategy_id"] not in strategies:
                raise StrategyLibraryError(f"variant {row['variant_id']} references missing strategy {row['strategy_id']}")
            variants[row["variant_id"]] = StrategyVariantNode(
                variant_id=row["variant_id"],
                display_name=row.get("display_name", row["variant_id"]),
                lineage_group=row.get("lineage_group"),
                strategy_id=row["strategy_id"],
                parent_id=row.get("parent_id"),
                strategy_version=row["strategy_version"],
                status=row["status"],
                hypothesis=row["hypothesis"],
                factor_set=dict(row.get("factor_set", {})),
                changed_factors=list(row.get("changed_factors", [])),
                core_logic=list(row.get("core_logic", [])),
                source_basis=list(row.get("source_basis", [])),
                risk_profile=dict(row.get("risk_profile", {})),
                iteration_notes=row.get("iteration_notes", ""),
                promotion_rules=list(row.get("promotion_rules", [])),
            )

        library = cls(families=families, strategies=strategies, variants=variants)
        library.validate()
        return library

    def validate(self) -> None:
        for variant in self.variants.values():
            if variant.parent_id is not None and variant.parent_id not in self.variants:
                raise StrategyLibraryError(f"variant {variant.variant_id} references missing parent {variant.parent_id}")
            self.lineage(variant.variant_id)

    def family_for_strategy(self, strategy_id: str) -> StrategyFamilyNode:
        strategy = self.strategies[strategy_id]
        return self.families[strategy.family_id]

    def variant(self, variant_id: str) -> StrategyVariantNode:
        if variant_id not in self.variants:
            raise StrategyLibraryError(f"unknown variant {variant_id}")
        return self.variants[variant_id]

    def lineage(self, variant_id: str) -> list[str]:
        seen: set[str] = set()
        chain: list[str] = []
        current = variant_id
        while current is not None:
            if current in seen:
                raise StrategyLibraryError(f"cycle detected at variant {current}")
            if current not in self.variants:
                raise StrategyLibraryError(f"variant {variant_id} references missing parent {current}")
            seen.add(current)
            chain.append(current)
            current = self.variants[current].parent_id
        return list(reversed(chain))


def ingest_leaderboard(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    leaderboard_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    run_id: str,
    strategy_version: str,
    data_snapshot_id: str,
    artifact_paths: dict[str, str] | None = None,
) -> IngestResult:
    library = StrategyLibrary.load(registry_path)
    rows = _read_leaderboard_rows(leaderboard_path)
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        _init_db(conn)
        inserted = 0
        for row in rows:
            variant_id = row.get("variant_id", "")
            parent_id = library.variants.get(variant_id).parent_id if variant_id in library.variants else None
            decision = "candidate" if _truthy(row.get("eligible")) else "rejected"
            metrics = _metrics_from_leaderboard_row(row)
            conn.execute(
                """
                insert or replace into strategy_runs (
                    run_id, strategy_version, variant_id, parent_variant_id, data_snapshot_id,
                    decision, score, right_tail_capture, loss_suppression, profit_factor,
                    max_drawdown, avg_win_loss_ratio, big_loss_overlap, signal_count,
                    paper_slippage_pnl, metrics_json, artifact_paths_json, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy_version,
                    variant_id,
                    parent_id,
                    data_snapshot_id,
                    decision,
                    _to_float(row.get("score")),
                    metrics["right_tail_capture"],
                    metrics["loss_suppression"],
                    metrics["profit_factor"],
                    metrics["max_drawdown"],
                    metrics["avg_win_loss_ratio"],
                    metrics["big_loss_overlap"],
                    metrics["signal_count"],
                    metrics["paper_slippage_pnl"],
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    json.dumps(artifact_paths or {"leaderboard": str(leaderboard_path)}, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return IngestResult(inserted_runs=inserted, db_path=str(db))


def compare_variants(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    parent_variant_id: str,
    child_variant_id: str,
) -> dict[str, Any]:
    library = StrategyLibrary.load(registry_path)
    parent = library.variant(parent_variant_id)
    child = library.variant(child_variant_id)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        parent_row = _latest_run(conn, parent_variant_id)
        child_row = _latest_run(conn, child_variant_id)
    finally:
        conn.close()
    return {
        "parent_variant_id": parent.variant_id,
        "child_variant_id": child.variant_id,
        "changed_factors": child.changed_factors,
        "right_tail_capture_delta": _delta(child_row, parent_row, "right_tail_capture"),
        "loss_suppression_delta": _delta(child_row, parent_row, "loss_suppression"),
        "profit_factor_delta": _delta(child_row, parent_row, "profit_factor"),
        "max_drawdown_delta": _delta(child_row, parent_row, "max_drawdown"),
        "avg_win_loss_ratio_delta": _delta(child_row, parent_row, "avg_win_loss_ratio"),
        "big_loss_overlap_delta": _delta(child_row, parent_row, "big_loss_overlap"),
        "signal_count_delta": _delta(child_row, parent_row, "signal_count"),
    }


def render_strategy_library_report(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    out_dir: str | Path = "docs/strategy_library",
) -> str:
    library = StrategyLibrary.load(registry_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest = _latest_runs_by_variant(db_path)
    index_text = _render_index(library, latest)
    index_path = out / "index.md"
    index_path.write_text(index_text, encoding="utf-8")
    for family_id, family in library.families.items():
        family_text = _render_family(library, family_id, latest)
        (out / f"{family.family_id}.md").write_text(family_text, encoding="utf-8")
    return str(index_path)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists strategy_runs (
            run_id text not null,
            strategy_version text not null,
            variant_id text not null,
            parent_variant_id text,
            data_snapshot_id text not null,
            decision text not null,
            score real,
            right_tail_capture real,
            loss_suppression real,
            profit_factor real,
            max_drawdown real,
            avg_win_loss_ratio real,
            big_loss_overlap real,
            signal_count real,
            paper_slippage_pnl real,
            metrics_json text not null,
            artifact_paths_json text not null,
            created_at text not null,
            primary key (run_id, variant_id)
        )
        """
    )
    conn.execute(
        "create index if not exists idx_strategy_runs_variant_created on strategy_runs (variant_id, created_at)"
    )


def _read_leaderboard_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metrics_from_leaderboard_row(row: dict[str, str]) -> dict[str, float | None]:
    return {
        "right_tail_capture": _first_float(row, ["right_tail_capture_score", "big_win_recall"]),
        "loss_suppression": _first_float(row, ["loss_suppression_score"]),
        "profit_factor": _first_float(row, ["validation_profit_factor", "profit_factor"]),
        "max_drawdown": _first_float(row, ["max_drawdown", "validation_max_drawdown"]),
        "avg_win_loss_ratio": _first_float(row, ["avg_win_loss_ratio"]),
        "big_loss_overlap": _first_float(row, ["big_loss_overlap"]),
        "signal_count": _first_float(row, ["validation_signals", "signal_count"]),
        "paper_slippage_pnl": _first_float(row, ["paper_slippage_pnl", "validation_net_pnl", "validation_realized_pnl_usdt"]),
    }


def _latest_run(conn: sqlite3.Connection, variant_id: str) -> sqlite3.Row:
    row = conn.execute(
        "select * from strategy_runs where variant_id = ? order by created_at desc limit 1",
        (variant_id,),
    ).fetchone()
    if row is None:
        raise StrategyLibraryError(f"no run found for variant {variant_id}")
    return row


def _latest_runs_by_variant(db_path: str | Path) -> dict[str, dict[str, Any]]:
    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _init_db(conn)
        rows = conn.execute(
            """
            select sr.* from strategy_runs sr
            join (
                select variant_id, max(created_at) as created_at
                from strategy_runs
                group by variant_id
            ) latest
            on sr.variant_id = latest.variant_id and sr.created_at = latest.created_at
            order by sr.strategy_version, sr.variant_id
            """
        ).fetchall()
        return {row["variant_id"]: dict(row) for row in rows}
    finally:
        conn.close()


def _render_index(library: StrategyLibrary, latest: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Strategy Library",
        "",
        "机器可读 registry 和 SQLite 实验账本是事实源；本文档由工具生成。",
        "",
        "## Strategy Tree",
        "",
    ]
    for family_id in sorted(library.families):
        family = library.families[family_id]
        lines.append(f"### {family.family_id}")
        lines.append("")
        for strategy in sorted(
            (row for row in library.strategies.values() if row.family_id == family_id),
            key=lambda row: row.strategy_id,
        ):
            lines.append(f"- `{strategy.strategy_id}` `{strategy.strategy_version}` `{strategy.status}`")
            for variant in sorted(
                (row for row in library.variants.values() if row.strategy_id == strategy.strategy_id),
                key=lambda row: row.variant_id,
            ):
                run = latest.get(variant.variant_id)
                metric = ""
                if run:
                    metric = (
                        f" latest_run=`{run['run_id']}` pf={_fmt(run['profit_factor'])} "
                        f"rt={_fmt(run['right_tail_capture'])} loss={_fmt(run['loss_suppression'])}"
                    )
                parent_label = _variant_display_name(library, variant.parent_id)
                lines.append(
                    f"  - `{family.family_id} / {strategy.strategy_id} / {variant.variant_id}` "
                    f"name=`{variant.display_name}` parent=`{parent_label}` status=`{variant.status}` "
                    f"lineage_group=`{variant.lineage_group or ''}` factors=`{', '.join(variant.changed_factors)}` "
                    f"core=`{', '.join(variant.core_logic)}`{metric}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_family(library: StrategyLibrary, family_id: str, latest: dict[str, dict[str, Any]]) -> str:
    family = library.families[family_id]
    lines = [f"# {family.name}", "", "## Variants", ""]
    for strategy in sorted(
        (row for row in library.strategies.values() if row.family_id == family_id),
        key=lambda row: row.strategy_id,
    ):
        lines.append(f"### {strategy.strategy_id}")
        lines.append("")
        lines.append(strategy.hypothesis)
        lines.append("")
        for variant in sorted(
            (row for row in library.variants.values() if row.strategy_id == strategy.strategy_id),
            key=lambda row: row.variant_id,
        ):
            run = latest.get(variant.variant_id)
            lines.append(f"- `{variant.display_name}` (`{variant.variant_id}`)")
            lines.append(f"  - parent: `{_variant_display_name(library, variant.parent_id)}`")
            lines.append(f"  - lineage: `{' -> '.join(_lineage_display_names(library, variant.variant_id))}`")
            lines.append(f"  - lineage_group: `{variant.lineage_group or ''}`")
            lines.append(f"  - changed_factors: `{', '.join(variant.changed_factors)}`")
            lines.append(f"  - core_logic: `{', '.join(variant.core_logic)}`")
            lines.append(f"  - hypothesis: {variant.hypothesis}")
            if variant.iteration_notes:
                lines.append(f"  - iteration_notes: {variant.iteration_notes}")
            if run:
                lines.append(
                    f"  - latest_run: `{run['run_id']}`, decision=`{run['decision']}`, "
                    f"profit_factor={_fmt(run['profit_factor'])}, max_drawdown={_fmt(run['max_drawdown'])}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _reject_duplicate_ids(rows: list[dict[str, Any]], key: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            raise StrategyLibraryError(f"duplicate {key}: {value}")
        seen.add(value)


def _require(row: dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        if field not in row:
            raise StrategyLibraryError(f"missing required field {field}")


def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _delta(child: sqlite3.Row, parent: sqlite3.Row, key: str) -> float | None:
    child_value = child[key]
    parent_value = parent[key]
    if child_value is None or parent_value is None:
        return None
    return float(child_value) - float(parent_value)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _variant_display_name(library: StrategyLibrary, variant_id: str | None) -> str:
    if not variant_id:
        return ""
    if variant_id not in library.variants:
        return variant_id
    return library.variants[variant_id].display_name


def _lineage_display_names(library: StrategyLibrary, variant_id: str) -> list[str]:
    return [_variant_display_name(library, row) for row in library.lineage(variant_id)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the LangLang strategy library")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--leaderboard", required=True)
    ingest_parser.add_argument("--run-id", required=True)
    ingest_parser.add_argument("--strategy-version", required=True)
    ingest_parser.add_argument("--data-snapshot-id", required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--out", default="docs/strategy_library")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--parent", required=True)
    compare_parser.add_argument("--child", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        library = StrategyLibrary.load(args.registry)
        print(f"OK: {len(library.families)} families, {len(library.strategies)} strategies, {len(library.variants)} variants")
        return 0
    if args.command == "ingest":
        result = ingest_leaderboard(
            registry_path=args.registry,
            leaderboard_path=args.leaderboard,
            db_path=args.db,
            run_id=args.run_id,
            strategy_version=args.strategy_version,
            data_snapshot_id=args.data_snapshot_id,
            artifact_paths={"leaderboard": args.leaderboard},
        )
        print(f"OK: inserted {result.inserted_runs} runs into {result.db_path}")
        return 0
    if args.command == "report":
        path = render_strategy_library_report(registry_path=args.registry, db_path=args.db, out_dir=args.out)
        print(f"OK: wrote {path}")
        return 0
    if args.command == "compare":
        print(
            json.dumps(
                compare_variants(
                    registry_path=args.registry,
                    db_path=args.db,
                    parent_variant_id=args.parent,
                    child_variant_id=args.child,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise StrategyLibraryError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
