# LangLang Trader

轻量模拟仓，用于把“浪浪”策略蒸馏结果放到接近真实交易的链路里观察。首版交易所主路径是 OKX USDT 永续，默认只跑 paper，不真实下单。

## 架构

- `langlang_trader/market_data.py`：统一行情接口，输出 `Candle`、`Ticker`、`OrderBook`。
- `langlang_trader/features.py`：把日线和多周期 K 线蒸馏成可复用特征快照。
- `langlang_trader/symbol_selection.py`：横截面筛币蒸馏，解释交割单当时为什么盯上某个合约，并给 fleet 提供候选币过滤。
- `langlang_trader/strategy.py`：保留 `rules_v01` 基线、`rules_langlang_v1` 增强版，并新增 `rules_langlang_v1_1` 完整闭环策略。
- `langlang_trader/strategy_source.py`：把 PDF 心得固化成章节、概念和规则覆盖矩阵。
- `langlang_trader/data_coverage.py`：按交易/合约/周期生成 K 线覆盖账本，缺失数据必须有原因。
- `langlang_trader/historical_patterns.py`：把交割单标签蒸馏成历史相似模式，为 v1.1 信号提供样本解释。
- `langlang_trader/risk.py`：把 `Signal` 转成 `OrderIntent`，处理仓位、杠杆、止损和滑点上限。
- `langlang_trader/execution/`：唯一可切换执行口，OKX/Binance paper executor、双交易所路由和 `OkxLiveExecutor` 保持同一接口。
- `langlang_trader/ledger.py`：SQLite 记录 signals、order_intents、orders、fills、positions、equity_snapshots、risk_events 和 raw_exchange_payloads；执行表带 `exchange` 维度，positions 按 `run_id / bot_id / exchange / symbol` 隔离。

## 运行

默认配置在 `configs/paper_okx.example.json`：

```bash
python3 -m langlang_trader.cli --config configs/paper_okx.example.json --once
```

`live_okx` 已留好接口，但有安全闸：必须同时满足 `mode=live`、`executor=live_okx`、`allow_live_orders=true`，并且环境变量存在 `OKX_API_KEY`、`OKX_API_SECRET`、`OKX_API_PASSPHRASE`。v0.1 默认不做真实下单。

## 多模拟舱

v0.2 支持多个 paper bot 共用一份行情、各自使用不同 `StrategyVariant`、并按 `run_id / bot_id / variant_id` 独立落账。

```bash
python3 -m langlang_trader.optimize --trades output/langlang_distill/standard_trades.csv --out output/fleet/latest
python3 -m langlang_trader.fleet_cli --config output/fleet/latest/selected_fleet_config.json --once
python3 -m langlang_trader.fleet_cli --config output/fleet/latest/selected_fleet_config.json --loop
```

当前 10 bot 策略树簇 paper 配置保存在 `configs/fleet/selected_fleet_config_langlang_10bot.json`。它使用固定 `run_id=langlang-paper-main-v1` 和持久化 SQLite 账本路径，重启后不会重置同一批 bot 的 paper 账本。默认仓位为浪浪 W 单位模型：用初始权益的 30% 作为活跃资金、分成 3 个 W，按浪位/市场季节/高低位纪律调整保证金；开仓数由 `output/langlang_v1_3/position_concurrency/position_concurrency_report.md` 的交割单并发统计反推，硬闸为单 bot 最多 3 个开仓、最多 3 个币、单笔名义不超过 5000 USDT、总入场名义不超过 25000 USDT：

```bash
python3 -m langlang_trader.fleet_cli \
  --config configs/fleet/selected_fleet_config_langlang_10bot.json \
  --loop
```

8 小时旧 run 中曾存在旧配置留下的超额持仓，所以后续因子评估应使用干净账本配置 `configs/fleet/selected_fleet_config_langlang_10bot_clean.json`。它保留同一 10 bot 因子和 W 仓位模型，但使用 `run_id=langlang-paper-clean-v1`、独立账本 `output/fleet/langlang_strategy_forest/clean/fleet_clean.sqlite3`，用于从 0 仓位开始观察 24-72 小时后再调整因子：

```bash
python3 -m langlang_trader.fleet_cli \
  --config configs/fleet/selected_fleet_config_langlang_10bot_clean.json \
  --loop \
  --interval-seconds 300

python3 -m langlang_trader.fleet_report \
  --ledger output/fleet/langlang_strategy_forest/clean/fleet_clean.sqlite3 \
  --run-id langlang-paper-clean-v1 \
  --out-dir output/fleet/langlang_strategy_forest/clean/reports
```

`fleet_report` 会写入 `latest.md` 和 `latest.json`，按 bot 汇总信号数、开仓/平仓、当前持仓、手续费、净权益变化和风控拒绝分布；权益只读取 `exchange=multi` 的汇总快照，避免 OKX/Binance 子账户快照重复计数。

优化器输出 `leaderboard.csv`、`selected_fleet_config.json`、`optimizer_report.md`。Fleet 支持 `paper_okx` 和 `paper_multi`；真实仓仍需要单独显式授权，不会批量开启 live。

`selected_fleet_config.json` 默认启用轻量筛币层：

```json
"selection": {
  "enabled": true,
  "top_n": 20,
  "min_score": 0.0,
  "min_daily_bars": 61
}
```

筛币层先用同一行情快照给所有合约打分，只让 Top N 候选进入各个 bot 的策略判断；已有仓位的止损检查不受筛币影响。

## 筛币蒸馏

可以用交割单和本地 OKX K 线 cache 生成“为什么选这个币”的独立报告：

```bash
python3 -m langlang_trader.symbol_selection \
  --trades output/langlang_distill/standard_trades.csv \
  --kline-cache output/langlang_distill/kline_cache \
  --out output/symbol_selection/latest \
  --top-n 20
```

输出包括 `symbol_selection_features.csv`、`symbol_selection_summary.csv`、`symbol_selection_report.md`。分析只使用入场日前已经完成的日线，避免把当天未收盘 K 线混入历史判断；缺失 K 线的合约会标记为 `selection_data_missing`，不混入硬结论。

## 浪浪交易法 v1.0

`rules_langlang_v1` 把主升浪、强趋势回调、突破回踩、顶部大分歧、弱势瀑布、追高过滤、结构止损和右尾持有计划落成可解释信号。v1.0 支持多空双向，signals 会记录 `regime / setup / filter_codes / decision_trace / historical_match_score`。

v1 优化器使用事件驱动 replay：信号先过风控，再用 `PaperExecutor` 生成 orders/fills/positions/equity；回放会执行结构止损、分批止盈、时间止损和趋势破坏退出。排行榜里 `validation_signals` 是实际可交易开仓数，`raw_validation_signals` 是原始策略触发密度；超过交易频率上限的候选会标记为 `event_replay_early_stop`，不会进入 selected fleet。

```bash
python3 -m langlang_trader.optimize \
  --strategy-version rules_langlang_v1 \
  --trades output/langlang_distill/standard_trades.csv \
  --out output/fleet/langlang_v1

python3 -m langlang_trader.optimize \
  --strategy-version rules_langlang_v1 \
  --trades output/langlang_distill/standard_trades.csv \
  --out output/fleet/langlang_v1_smoke \
  --max-variants 12

python3 -m langlang_trader.fleet_cli \
  --config output/fleet/langlang_v1/selected_fleet_config.json \
  --once
```

v1.0 仍是 paper/replay 验证路径；真实下单接口保留，但不会因为优化器或 fleet 配置自动开启 live。

## 浪浪交易法 v1.1

`rules_langlang_v1_1` 在 v1 的基础上补齐 PDF 概念覆盖、上方空间过滤、第一次 10x 后高位追价过滤、大分歧后底部抬升确认、连续止损簇、`W/nW` 风险单位和历史相似样本支持。非探索 bot 如果没有足够历史相似样本，会直接 `skip:no_historical_support`，不会生成可交易信号。

先生成 v1.1 完成度产物：

```bash
python3 -m langlang_trader.v1_1_artifacts --out output/langlang_v1_1
```

主要输出：

- `output/langlang_v1_1/pdf_craft/strategy_text.md`
- `output/langlang_v1_1/pdf_craft/strategy_sections.json`
- `output/langlang_v1_1/market_data_coverage.csv`
- `output/langlang_v1_1/historical_patterns.csv`
- `output/langlang_v1_1/symbol_selection_context.csv`
- `output/langlang_v1_1/v1_1_completion_report.md`

再跑 v1.1 事件回放和 paper fleet 配置：

```bash
python3 -m langlang_trader.optimize \
  --strategy-version rules_langlang_v1_1 \
  --trades output/langlang_distill/standard_trades.csv \
  --out output/fleet/langlang_v1_1 \
  --top-n 10

python3 -m langlang_trader.fleet_cli \
  --config output/fleet/langlang_v1_1/selected_fleet_config_v1_1.json \
  --once
```

v1.1 生成的 fleet 配置会给 `market_data.max_fetch_workers=8`，用于受控并发拉取 OKX 公共 K 线；如果只想做上线前 smoke，可以先限制 symbol 宇宙：

```bash
python3 -m langlang_trader.fleet_cli \
  --config output/fleet/langlang_v1_1/selected_fleet_config_v1_1.json \
  --once \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
```

v1.1 的输出文件带 `_v1_1` 后缀，不覆盖 v1/v0.2 结果。当前全量 replay 已能闭环生成排行榜，但 paper 前要重点看验证期净收益、最大回撤和大亏重叠，不能只看解释完整度。

## 浪浪全市场筛币 v1.2

`rules_langlang_v1_2` 复用 v1.1 的入场、风控、退出和解释闭环，但把筛币升级成 OKX 全市场 `USDT-SWAP` 可执行池，并拆成两张独立榜：`long_main_wave` 主升浪多头榜和 `short_waterfall` 瀑布空头榜。BTC/ETH 只作为市场环境参考，不进入山寨候选榜。Fleet 中 long bot 只消费多头榜，short bot 只消费空头榜，探索 bot 可观察双榜但会记录来源。

当前 universe 已接入 Binance USDT-M 永续：`okx_binance_usdt_swap_observe` 会同时拉 OKX 和 Binance，`observed_symbols` 包含 Binance 独有合约，`symbols` 仍保留 OKX 可执行池用于兼容旧逻辑。v1.2 的 `paper_multi` 执行会通过 `ExecutionRouter` 路由：OKX/Binance 共有币默认走 Binance paper，OKX 独有走 OKX paper，Binance 独有走 Binance paper；策略、筛币和风控仍只处理标准 `OrderIntent`。

先生成离线筛币产物，用本地 K 线缓存解释交割单里“为什么会盯这个币”：

```bash
python3 -m langlang_trader.v1_2_artifacts \
  --trades output/langlang_distill/standard_trades.csv \
  --kline-cache output/langlang_distill/kline_cache \
  --out output/langlang_v1_2 \
  --live-universe \
  --universe-provider okx_binance
```

主要输出：

- `output/langlang_v1_2/universe_snapshot.json`
- `output/langlang_v1_2/selection_long_leaderboard.csv`
- `output/langlang_v1_2/selection_short_leaderboard.csv`
- `output/langlang_v1_2/symbol_selection_context_v1_2.csv`

再生成 v1.2 paper fleet 配置：

```bash
python3 -m langlang_trader.optimize \
  --strategy-version rules_langlang_v1_2 \
  --trades output/langlang_distill/standard_trades.csv \
  --kline-cache output/langlang_distill/kline_cache \
  --out output/fleet/langlang_v1_2 \
  --top-n 10
```

v1.2 的 `selected_fleet_config_v1_2.json` 会设置：

```json
"universe": { "mode": "okx_binance_usdt_swap_observe", "provider": "okx_binance" },
"execution": { "mode": "paper", "exchange": "multi", "executor": "paper_multi" },
"routing": { "shared_symbol_policy": "binance_first" },
"selection": { "style": "dual_board", "long_top_n": 30, "short_top_n": 20 },
"market_data": { "symbols": [] }
```

正式 paper 时会实时拉 OKX 全市场 live USDT 永续 + Binance USDT-M 永续观察池；OKX 优先拉行情，OKX 没有的 symbol 自动 fallback 到 Binance Futures 行情。上线前 smoke 可以临时限制 symbol，避免一次拉全市场：

```bash
python3 -m langlang_trader.fleet_cli \
  --config output/fleet/langlang_v1_2/selected_fleet_config_v1_2.json \
  --once \
  --symbols BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,DOGE-USDT-SWAP
```

真实下单仍不属于 v1.2 默认路径；`fleet` 在 v1.2 默认使用 `paper_multi`，但不会启用 Binance/OKX live。

## 测试

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q langlang_trader tests
```
