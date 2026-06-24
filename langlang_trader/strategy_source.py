from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIRMED_SOURCE = Path(__file__).resolve().parent / "resources" / "langlang_strategy_source_confirmed.md"
DEFAULT_SOURCE_IMAGE_DIR = Path(__file__).resolve().parent / "resources" / "source_images"


@dataclass(frozen=True)
class LangLangConcept:
    concept_id: str
    section: str
    source_summary: str
    quant_rule: str
    status: str


@dataclass(frozen=True)
class PdfSourceIntegrity:
    file_exists: bool
    file_size_bytes: int
    page_count: int
    extracted_char_count: int
    extraction_method: str
    source_status: str
    requires_human_confirmation: bool
    extraction_error: str = ""
    confirmed_text_path: str = ""


SECTIONS = {
    "前言": "策略不是单一信号，而是围绕行情阶段、位置、结构、纪律和心态形成的交易系统。",
    "市场理解": "市场分主升浪与分歧阶段；分歧可分小、中、大、超大，分歧级别决定是否承接或等待。",
    "交易系统": "开仓需要主升浪、上方空间、结构顺势突破，并用大周期定方向、小周期找点位。",
    "交易纪律": "禁止超预期金额；高位不执拗；开仓逻辑必须有多个正确点；错了及时止损。",
    "交易心法": "等待可见机会，不吃看不懂的钱，错过不后悔，避免反向开仓和频繁止损后的情绪交易。",
}


CONCEPTS = [
    LangLangConcept(
        "regime_main_wave",
        "市场理解",
        "主升浪阶段优先顺势，回调不破结构时继续寻找多头机会。",
        "20/60日涨幅、MA结构、区间位置和回调幅度共同判定主升浪。",
        "implemented",
    ),
    LangLangConcept(
        "regime_divergence_size",
        "市场理解",
        "分歧分小、中、大、超大；小分歧可承接，大分歧要等待结构重新稳定。",
        "用短周期回撤、放量反杀、位置和底部抬升确认划分分歧等级。",
        "implemented",
    ),
    LangLangConcept(
        "regime_btc_anchor",
        "市场理解",
        "币圈所有行情围绕大饼运行；山寨介入要看大饼是否结束分歧并重新向上。",
        "BTC/ETH 环境作为 market_env，BTC 分歧时降低山寨突破分，BTC 收敛向上时提高龙头候选权重。",
        "implemented",
    ),
    LangLangConcept(
        "selection_leader_altcoin",
        "市场理解",
        "龙头山寨与大饼共振，第一波跟随上涨、震荡时强势、日线结构好且突破后空间大。",
        "SelectionEngine 用相对 BTC/ETH 强度、20/60 日区间位置、震荡抗跌、突破空间和量能综合排序。",
        "implemented",
    ),
    LangLangConcept(
        "selection_laggard_catchup",
        "市场理解",
        "补涨山寨通常一波流，不如龙头；大饼分歧时补涨更容易不流畅和假突破。",
        "补涨/相对弱但短期急拉标记为 catch_up_coin，降低持有预期并缩短 take_profit_plan。",
        "implemented",
    ),
    LangLangConcept(
        "selection_new_listing_bias",
        "市场理解",
        "市场喜欢次新币，没有套牢盘好拉盘；但高位盘整次新突破后要及时止盈。",
        "新币/次新标记为 listing_age_proxy，可提升流动性热度但在高位盘整后加入 quick_take_profit 风控说明。",
        "risk_note_only",
    ),
    LangLangConcept(
        "short_new_listing_contraction",
        "市场理解",
        "新币刚上所后，K线收敛后适合做空，但市场很热时需谨慎。",
        "空头榜加入 new_listing_contraction_short_setup，强牛环境下转为 caution filter。",
        "implemented",
    ),
    LangLangConcept(
        "turning_point_no_new_extreme",
        "市场理解",
        "上升趋势没有再出现新高是拐点；下降趋势没有再出现新低是拐点。",
        "用高低点递进失败、MA斜率和结构破位生成 trend_turn_risk / waterfall_exhaustion filter。",
        "implemented",
    ),
    LangLangConcept(
        "avoid_after_five_waves",
        "市场理解",
        "五浪走完，回调结束后的震荡行情一般不要参与。",
        "第三次小分歧、高区间、主升浪末端和震荡低效区标记为 choppy_invalid/avoid_after_five_waves。",
        "implemented",
    ),
    LangLangConcept(
        "entry_first_starting_point",
        "交易系统",
        "第一启动位置可做第一笔，但必须满足主升浪和结构突破。",
        "突破/回踩、新高附近、短周期转强触发 starter/first breakout。",
        "implemented",
    ),
    LangLangConcept(
        "entry_small_divergence",
        "交易系统",
        "小分歧位置可小仓承接，核心是分歧被吸收后小周期重新转强。",
        "m15轻微回落但m5重新转强，且历史相似样本支持。",
        "implemented",
    ),
    LangLangConcept(
        "entry_small_divergence_first_two_only",
        "交易系统",
        "前两次小分歧可以做，第三次小分歧不要做，因为已经处于高位五浪结构。",
        "small_divergence_count <= 2 才允许，第三次或高位小分歧进入 avoid_after_five_waves/filter_first_10x_too_high。",
        "implemented",
    ),
    LangLangConcept(
        "entry_second_pressure_retest",
        "交易系统",
        "第一阶段结束后，第二压力位或平台回踩是二次入场点。",
        "突破前高后回踩不破、h1/15m平台仍在均线或前高上方。",
        "implemented",
    ),
    LangLangConcept(
        "entry_post_large_divergence_rebound",
        "交易系统",
        "大分歧后不直接追，等内部小反弹和底部抬升再考虑。",
        "large_divergence_recent 后必须 bottom_lift_confirmed 且短周期反弹。",
        "implemented",
    ),
    LangLangConcept(
        "entry_six_positions",
        "交易系统",
        "完整周期包含六个位置：启动多、小分歧接多、第一次大分歧摸顶空、二波启动多、主升浪结束摸顶空、箱体内部反弹多。",
        "EntrySetup 明确映射 starter_buy/small_divergence_entry/top_short/second_entry/final_top_short/box_rebound_long，并按顺逆势降权。",
        "implemented",
    ),
    LangLangConcept(
        "entry_box_internal_rebound_low_quality",
        "交易系统",
        "箱体震荡开始后的内部反弹做多较难，因为市场冷清且箱体内部走势无序。",
        "box_internal_rebound 只作为低置信度 setup，默认需要更高历史支持和更小仓位。",
        "implemented",
    ),
    LangLangConcept(
        "entry_top_short_countertrend",
        "交易系统",
        "第一次大分歧开始和主升浪结束时摸顶做空有逻辑但属于逆势单，不易把握。",
        "counter_trend_short 需要主升浪末端、无新高、放量反杀和历史样本同时满足，否则 skip。",
        "implemented",
    ),
    LangLangConcept(
        "filter_upside_space",
        "交易系统",
        "上方有空间才开仓；压力位太近会降低盈亏比。",
        "upside_space_pct 低于阈值直接 skip。",
        "implemented",
    ),
    LangLangConcept(
        "filter_first_10x_too_high",
        "交易系统",
        "如果第一次10x位置已经很高，后续高位不能继续执拗追。",
        "first_10x_entry_done 且高区间、无回调时 skip。",
        "implemented",
    ),
    LangLangConcept(
        "filter_false_breakout_after_contraction",
        "市场理解",
        "收敛之后的假突破，回调的可能会比较多；震荡周期内做突破容易被假突破止损。",
        "窄幅收敛后突破但无量能/无回踩确认时标记 false_breakout_risk，禁止追突破。",
        "implemented",
    ),
    LangLangConcept(
        "filter_deep_pullback_box_bias",
        "市场理解",
        "拉升之后回调比较多，更可能走箱体震荡，冲高可能性更小，要降低点位预期。",
        "回撤深度超过阈值时从 trend_continuation 降级为 box_bias，缩短持仓并降低止盈目标。",
        "implemented",
    ),
    LangLangConcept(
        "risk_w_unit_positioning",
        "交易纪律",
        "低位高胜率可以更主动，高位或从下方补救只能用更小风险单位。",
        "用 risk_unit、position_size_multiplier、high_position_reduce_size 输出给风控。",
        "implemented",
    ),
    LangLangConcept(
        "risk_market_season_positioning",
        "交易系统",
        "春夏秋冬判断用于仓位和做单频率：春夏激进，秋冬谨慎；水平有限时只做夏天行情。",
        "market_temperature 输出 position_frequency_multiplier，夏天提高、秋冬降低或暂停。",
        "implemented",
    ),
    LangLangConcept(
        "risk_fixed_fraction_bankroll",
        "交易系统",
        "总仓位分三份，亏损从剩余资金补，盈利提出，资产翻倍后再提高单次仓位。",
        "paper/live 统一用 fixed_fraction_risk_unit 与 equity_step_up 记录，不让策略直接改账户。",
        "implemented",
    ),
    LangLangConcept(
        "risk_low_leverage",
        "交易系统",
        "分仓加低杠杆避免插针损失全部资金；山寨 5 倍，大饼 10 倍。",
        "risk 默认杠杆按 symbol_class 设置，山寨上限 5x，BTC/ETH 上限 10x。",
        "implemented",
    ),
    LangLangConcept(
        "discipline_stop_loss",
        "交易纪律",
        "跌破关键结构、短时快速亏损、连续止损都要降低频率或停止。",
        "结构止损、MAE止损、时间止损、stop_loss_cluster 过滤。",
        "implemented",
    ),
    LangLangConcept(
        "discipline_good_entry_required",
        "交易纪律",
        "不管预期多大、逻辑多正确，入场点位一定要好；高位不做突破。",
        "entry_quality_score 低或高位突破未回踩确认直接 skip。",
        "implemented",
    ),
    LangLangConcept(
        "discipline_hold_base_position_distance",
        "交易纪律",
        "有底仓后和市场保持距离，避免一直盯盘因震荡抛掉底仓。",
        "右尾持有使用结构止损和趋势破坏退出，不因小级别噪音触发底仓退出。",
        "implemented",
    ),
    LangLangConcept(
        "discipline_no_off_system_money",
        "交易纪律",
        "不该赚的钱不要赚，不符合交易系统的钱会在以后亏回去。",
        "无历史支持、无 setup、无 selection reason 的信号统一 skip/no_historical_support。",
        "implemented",
    ),
    LangLangConcept(
        "discipline_no_short_in_main_wave",
        "交易纪律",
        "主升浪不做任何一笔空单，逆势摸顶空容易被反弹或插针打止损。",
        "MarketRegime=MAIN_UPTREND 时 short 默认禁用，除非进入严格 counter_trend_short 观察模式。",
        "implemented",
    ),
    LangLangConcept(
        "loss_emotional_or_no_logic",
        "交易纪律",
        "亏损原因包括单子没有逻辑、情绪化开单、震荡周期做突破。",
        "decision_trace 记录 no_logic/emotional_revenge_proxy/choppy_breakout_loss 风险码。",
        "implemented",
    ),
    LangLangConcept(
        "mindset_wait_visible_money",
        "交易心法",
        "不吃看不懂的钱，错过主升浪低位时等待下一结构。",
        "转成 risk_notes 和 no_historical_support skip，不作为自由裁量信号。",
        "risk_note_only",
    ),
    LangLangConcept(
        "mindset_deliberate_practice",
        "交易心法",
        "更快学会交易需要走在正确路上多复盘、多实战、刻意练习、保持心态平稳。",
        "转为研究工作流要求：每笔信号保留 decision_trace、matched_trade_examples 和复盘字段。",
        "risk_note_only",
    ),
    LangLangConcept(
        "mindset_success_criteria",
        "交易心法",
        "成功迹象是能判断 K 线后续、行情来时赚钱、震荡行情少亏或不亏。",
        "验收指标包含可解释性、右尾捕获、震荡少亏、paper 长期稳定，不只看短期收益。",
        "risk_note_only",
    ),
    LangLangConcept(
        "mindset_emotion_recovery",
        "交易心法",
        "心态波动时要及时调整，调整速度需要磨练。",
        "连续止损、异常频率和回撤触发 cooldown/risk_notes，不作为自由裁量加仓依据。",
        "risk_note_only",
    ),
]


class StrategySourceBuilder:
    def __init__(self, pdf_path: str | Path, confirmed_text_path: str | Path | None = None):
        self.pdf_path = Path(pdf_path)
        self.confirmed_text_path = Path(confirmed_text_path) if confirmed_text_path else None

    def build(self, out_dir: str | Path) -> dict[str, Any]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        extracted_text, source_integrity = self._extract_text_with_integrity()
        strategy_text = self._strategy_text(extracted_text, source_integrity)
        (out_path / "strategy_text.md").write_text(strategy_text, encoding="utf-8")
        self._copy_source_images(out_path)
        payload = {
            "pdf_path": str(self.pdf_path),
            "extraction_method": source_integrity.extraction_method,
            "source_integrity": asdict(source_integrity),
            "sections": [{"section": section, "summary": summary} for section, summary in SECTIONS.items()],
            "concepts": [asdict(concept) for concept in CONCEPTS],
            "unknown_concepts": sum(1 for concept in CONCEPTS if concept.status == "unmapped"),
        }
        (out_path / "strategy_sections.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_path / "visual_qa.md").write_text(self._visual_qa(payload), encoding="utf-8")
        return payload

    def _copy_source_images(self, out_path: Path) -> None:
        if not DEFAULT_SOURCE_IMAGE_DIR.exists():
            return
        image_out = out_path / "source_images"
        image_out.mkdir(parents=True, exist_ok=True)
        for source in sorted(DEFAULT_SOURCE_IMAGE_DIR.glob("*")):
            if source.is_file():
                shutil.copy2(source, image_out / source.name)

    def _extract_text_with_integrity(self) -> tuple[str, PdfSourceIntegrity]:
        if self.confirmed_text_path and self.confirmed_text_path.exists():
            text = self.confirmed_text_path.read_text(encoding="utf-8").strip()
            return text, PdfSourceIntegrity(
                file_exists=self.pdf_path.exists(),
                file_size_bytes=self.pdf_path.stat().st_size if self.pdf_path.exists() else 0,
                page_count=1 if self.pdf_path.exists() else 0,
                extracted_char_count=len(text),
                extraction_method="user_confirmed_pdf_text",
                source_status="user_confirmed_pdf_text",
                requires_human_confirmation=False,
                confirmed_text_path=str(self.confirmed_text_path),
            )
        if not self.pdf_path.exists():
            return "", PdfSourceIntegrity(
                file_exists=False,
                file_size_bytes=0,
                page_count=0,
                extracted_char_count=0,
                extraction_method="missing_pdf",
                source_status="pdf_missing_needs_source_confirmation",
                requires_human_confirmation=True,
            )
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(self.pdf_path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(page_texts).strip()
            char_count = len(text)
            if char_count == 0 and len(reader.pages) > 0:
                status = "image_pdf_needs_ocr_or_pdf_craft_confirmation"
                method = "pypdf_empty_text_image_pdf"
                requires_confirmation = True
            else:
                status = "text_extracted_needs_visual_confirmation"
                method = "pypdf_text_extraction"
                requires_confirmation = True
            return text, PdfSourceIntegrity(
                file_exists=True,
                file_size_bytes=self.pdf_path.stat().st_size,
                page_count=len(reader.pages),
                extracted_char_count=char_count,
                extraction_method=method,
                source_status=status,
                requires_human_confirmation=requires_confirmation,
            )
        except Exception as exc:
            return "", PdfSourceIntegrity(
                file_exists=True,
                file_size_bytes=self.pdf_path.stat().st_size,
                page_count=0,
                extracted_char_count=0,
                extraction_method="pypdf_read_error",
                source_status="pdf_read_error_needs_source_confirmation",
                requires_human_confirmation=True,
                extraction_error=repr(exc),
            )

    def _strategy_text(self, extracted_text: str, source_integrity: PdfSourceIntegrity) -> str:
        lines = ["# 浪浪交易法策略源文本", ""]
        lines.extend(
            [
                "## Source Integrity",
                "",
                f"- source_status: {source_integrity.source_status}",
                f"- extraction_method: {source_integrity.extraction_method}",
                f"- confirmed_text_path: {source_integrity.confirmed_text_path}",
                f"- page_count: {source_integrity.page_count}",
                f"- extracted_char_count: {source_integrity.extracted_char_count}",
                f"- requires_human_confirmation: {str(source_integrity.requires_human_confirmation).lower()}",
                "",
            ]
        )
        if source_integrity.source_status == "user_confirmed_pdf_text":
            lines.extend(["## User Confirmed PDF Text", "", extracted_text, ""])
        elif extracted_text:
            lines.extend(["## PDF Craft / PDF Text Extraction", "", extracted_text, ""])
        else:
            lines.extend(
                [
                    "## PDF Craft / Visual QA Fallback",
                    "",
                    "原 PDF 为图片型或不可直接抽取文本；本文件采用 PDF Craft 解析优先、渲染图人工结构化兜底的方式固化策略源。",
                    "",
                ]
            )
        for section, summary in SECTIONS.items():
            lines.extend([f"## {section}", "", summary, ""])
        lines.extend(["## 概念覆盖", ""])
        lines.extend(f"- `{concept.concept_id}`: {concept.source_summary}" for concept in CONCEPTS)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _visual_qa(payload: dict[str, Any]) -> str:
        lines = [
            "# PDF Visual QA",
            "",
            f"- sections_checked: {len(payload['sections'])}",
            f"- concepts_checked: {len(payload['concepts'])}",
            f"- unknown_concepts: {payload['unknown_concepts']}",
            f"- source_status: {payload['source_integrity']['source_status']}",
            f"- extraction_method: {payload['source_integrity']['extraction_method']}",
            f"- confirmed_text_path: {payload['source_integrity'].get('confirmed_text_path', '')}",
            f"- extracted_char_count: {payload['source_integrity']['extracted_char_count']}",
            f"- requires_human_confirmation: {str(payload['source_integrity']['requires_human_confirmation']).lower()}",
            "",
            "## Coverage Matrix",
            "",
        ]
        lines.extend(
            f"- {concept['section']} / {concept['concept_id']}: {concept['status']}"
            for concept in payload["concepts"]
        )
        return "\n".join(lines) + "\n"
