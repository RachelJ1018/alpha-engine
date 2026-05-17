# Alpha Engine — Evolution Plan v2

目标：从"每天扫描股票的打分工具"进化为"可以辅助投资决策的事件驱动研究助手"。

**当前状态：** 数据管道稳定，Evaluation 体系已完成，score / tier 已完成第一轮实证校准。score ≥52 + gates 显示正 alpha，新 ACTIONABLE 分层初步有效，但仍需要 forward validation。

## 核心原则

- 少而精 > 多而杂
- empirical threshold > 理论满分阈值
- score 是筛选器，不是自动交易指令
- risk penalty 和 position sizing 分离
- 人工判断是最终门
- 小仓实盘可以作为 controlled experiment，但不能视为策略已被证明

---

## Phase 0 — 基础系统与评估体系【已完成】

### 0.1 数据与信号管道
- [x] 新闻采集 → 价格快照 → 分析 → 报告 → 通知
- [x] t1 / t3 / t5 outcome tracking
- [x] paper trade tracking：stop / target / T5 exit
- [x] signal_outcomes 持续记录 resolved signals
- [x] Streamlit Evaluation tab
- [x] High Conviction / Action 字段接入报告和 UI

### 0.2 Evaluation 体系补齐
- [x] Score bucket return
- [x] Benchmark-adjusted return
- [x] De-duplicated event return
- [x] R-multiple analysis
- [x] Win rate confidence interval
- [x] Component correlation report
- [x] Promoted signal quality report
- [x] False upgrade diagnosis
- [x] Volume × MarketConf matrix
- [x] Empirical threshold backtest
- [x] Tier backtest

### 0.3 当前关键数据

| 指标 | 当前结果 |
|------|---------|
| Resolved signals | 124 |
| Baseline win rate | 62.9% |
| Baseline avg t5 | +1.34% |
| Baseline avg alpha | +0.46% |
| Baseline avg R | +0.36R |
| score ≥52 n | 62 |
| score ≥52 win rate | 66.1% |
| score ≥52 avg t5 | +1.87% |
| score ≥52 avg alpha | +1.02% |
| score ≥52 avg R | +0.47R |
| score ≥52 worst t5 | -4.92% |
| New ACTIONABLE n | 7 |
| New ACTIONABLE win rate | 71.4% |
| New ACTIONABLE avg t5 | +2.56% |

**当前判断：**
- score ≥52 是目前最合理的 empirical tradeable cutoff。
- ACTIONABLE 分层初步有效，但 n=7，仍是 promising，不是 proven。

---

## Phase 1 — Scoring 校准与降噪【已完成】

目标：让 score 从"叙事完整度"变成"弱正向 alpha predictor"。

### 1.1 RiskPenalty 重写
- [x] RSI 逻辑改成 event-type-aware
- [x] LONG event-driven + high RSI 不再自动惩罚
- [x] SHORT + RSI >70 不再误判为 counter-trend
- [x] choppy 不再扣 score，改由 position_size_mult 处理
- [x] high ATR 在 confirmed post-earnings setup 中不再强扣 score
- [x] high ATR 风险转移到 position sizing

**结论：** RiskPenalty rewrite: PASS
修正了 AMD / UNH / TSLA 这类强 earnings signal 被错误惩罚的问题。

### 1.2 Gap bonus gate
- [x] Low-volume earnings gap 不再自动加分
- [x] SHORT gap-down 不再直接给 bonus
- [x] Neutral + weak volume 降低或取消 bonus
- [x] weak confirmation 下限制 RP reduction

**结论：** Gap bonus gate: PASS
减少了 low-volume / weak-confirmation false upgrades。

### 1.3 RelOpp strategy-aware rewrite
- [x] macro_watch 降分
- [x] sympathy_play 降分
- [x] event_short 改成 resistance / near-ATH 逻辑
- [x] post_earnings_drift 不再被 near-ATH 直接归零
- [x] mean_reversion_long 保留 distance discount 逻辑

**结论：** RelOpp rewrite: PASS as calibration, not as score booster.
RelOpp 的价值主要是压低噪音 bucket，而不是制造 ACTIONABLE。

### 1.4 Empirical threshold 替代旧理论阈值

旧系统问题：ACTIONABLE threshold = 77，历史 max score ≈ 65，导致 ACTIONABLE 永远不可达。

- [x] 废弃旧 77 / 62 阈值
- [x] 新 empirical cutoff: score ≥52
- [x] macro_watch / sympathy_play / opinion_watch → structural IGNORE
- [x] EventEdge guard 保留
- [x] position_size_mult 纳入 tier gate

---

## Phase 2 — Forward Validation & Tier Quality【当前阶段】

目标：验证新 tier 是否在未来数据中继续有效，而不是只在历史样本里有效。

### 2.1 上线后核心监控

每日 / 每周追踪：
- [ ] WATCHLIST count
- [ ] ACTIONABLE count
- [ ] ACTIONABLE vs WATCHLIST vs IGNORE 的 avg t5
- [ ] ACTIONABLE vs WATCHLIST vs IGNORE 的 avg alpha
- [ ] ACTIONABLE vs WATCHLIST vs IGNORE 的 avg R
- [ ] ACTIONABLE worst loss
- [ ] HIT_TARGET / HIT_STOP / T5_EXIT
- [ ] score ≥52 rolling 30-day alpha
- [ ] score ≥52 rolling 30-day R-multiple

**通过标准：**
- ACTIONABLE avg_alpha > WATCHLIST
- ACTIONABLE avg_R > WATCHLIST
- ACTIONABLE worst loss 可控
- score ≥52 rolling avg_alpha > 0
- score ≥52 rolling avg_R > baseline

**失败信号：**
- ACTIONABLE 不如 WATCHLIST
- score ≥52 alpha 变负
- HIT_STOP rate 明显上升
- worst loss 明显扩大
- macro_watch / sympathy_play 重新进入高优先级

### 2.2 ACTIONABLE 先视为 High Priority Candidate

历史 ACTIONABLE 结果：n=7, win=71.4%, avg_t5=+2.56%，losses were small: -0.03%, -0.45%。但样本仍太小。

- [x] 新 ACTIONABLE 规则上线
- [ ] UI / 文案中更准确地理解为 ACTIONABLE_CANDIDATE 或 HIGH_PRIORITY
- [ ] 累积 20–30 个 forward ACTIONABLE 后重新评估

**当前判断：** ACTIONABLE rule is promising, not proven.

### 2.3 50–52 区间增加 Near Watchlist

MONITOR 中有一些边缘好信号，但没有证据说明应该把 WATCHLIST 阈值降到 50。

- [ ] 保留 WATCHLIST threshold = 52
- [ ] 新增 NEAR_WATCHLIST 标记：50 ≤ score < 52
- [ ] 显示 missed-by reason：score missed by X / low volume / low MarketConf / bucket not alpha-positive / position_size_mult too low / EventEdge 不足

**原则：** 不降低阈值，但允许人工复核 near-watchlist。

### 2.4 Position-size-adjusted portfolio backtest

因为 choppy / high ATR 已从 score penalty 转移到 position_size_mult，需要验证 portfolio-level 效果。

- [ ] Equal-weight return
- [ ] Position-size-adjusted return
- [ ] Max drawdown
- [ ] Volatility
- [ ] Worst 5 trades contribution
- [ ] Exposure by regime
- [ ] Exposure by strategy_bucket
- [ ] Exposure by ticker / sector

**通过标准：**
- position-sized drawdown < equal-weight drawdown
- return 不明显下降
- worst trade impact 下降

### 2.5 earn_strength forward tracking

历史 trade_candidates 没有 earn_strength，所以 condition b 无法完整回测。

- [x] 新信号完整记录 earn_strength（已写入 trade_candidates）
- [ ] 单独追踪 earn_strength ≥3 的表现
- [ ] 看 earn_strength 是否真的提升 ACTIONABLE quality

指标：earn_strength ≥3: n / win rate / avg t5 / avg alpha / avg R / worst loss

---

## Phase 3 — Setup-specific Alpha Validation【1–3 个月】

目标：不再强求一个 global score 解释所有 setup，而是按 strategy_bucket 验证 edge。

### 3.1 post_earnings_drift

当前最有希望的 setup。

已发现的强组合：volume_ratio 1.0–1.5，MarketConf 10–15

历史表现：n=6, win=83.3%, avg_t5=+5.76%, avg_alpha=+5.47%, avg_R=+1.09

- [ ] Forward 追踪 post_earnings_drift
- [ ] 分开记录 gap-up continuation vs gap-down reversal
- [ ] 记录 earnings strength / volume_ratio bucket / MarketConf bucket
- [ ] 记录 alpha vs benchmark / sector ETF
- [ ] 30+ forward samples 后重新验证

### 3.2 event_short

SHORT 不再完全禁用，但必须更严格。

- [ ] 单独追踪 event_short
- [ ] 只允许 high-confirmation event_short
- [ ] 避免 low-volume gap-down chasing
- [ ] 分开看 earnings short / non-earnings short

**当前原则：** SHORT 不是禁用，而是必须更严格 gated。

### 3.3 relative_strength_long

UNH 等样本显示有潜力，但样本太小。

- [ ] 暂不加入 alpha-positive bucket
- [ ] 单独追踪 relative_strength_long
- [ ] n ≥20 后评估是否加入 `_ALPHA_POS`

### 3.4 macro_watch / sympathy_play / opinion_watch

当前作为 structural IGNORE。

- [x] macro_watch → IGNORE
- [x] sympathy_play → IGNORE
- [x] opinion_watch → IGNORE
- [ ] 只有未来独立证明 positive alpha 后才重新开放

---

## Phase 4 — EventEdge 质量提升【3–6 个月】

目标：提高事件识别质量，而不是机械提高分数。

**当前发现：** EventEdge 常在 17–20；强 earnings signal 也不一定拿到接近 25 的分数。不要直接加权，先做 EventEdge audit。

### 4.1 EventEdge bucket report

分 EE <12 / 12–16 / 16–20 / 20+ 四档，每档看：n / win rate / avg_t5 / avg_alpha / avg_R / HIT_STOP / worst loss。

- [ ] 只有 EE 高分单调更好，才考虑提高权重

### 4.2 增强 earnings event quality

未来可以纳入：EPS surprise / Revenue surprise / Guidance raise|cut / Margin expansion / Analyst revision / Management tone / Multi-source confirmation / Premarket reaction quality

**目标：** EventEdge 从"新闻存在"升级为"事件质量"。

---

## Phase 5 — 实盘验证路径

### Phase 5A — 极小仓 live experiment【现在可开始】

目标：验证真实执行，而不是证明策略已经成熟。

**前置条件：**
- [ ] 只做 ACTIONABLE_CANDIDATE
- [ ] 每笔必须人工确认
- [ ] 不做自动交易
- [ ] 不加仓、不摊平、不主观延长持有
- [ ] 严格记录真实 entry / exit / slippage / reason
- [ ] Paper result 和 live result 分开统计

**仓位规则：** 每笔账户风险 ≤ 0.1%–0.25%，每次最多 1 笔，最多 1–2 个高相关仓位

**允许开始的理由：** score ≥52 已显示正 alpha；ACTIONABLE historical n=7 表现良好；但样本小，因此只能极小仓验证。

### Phase 5B — 小仓验证【新增 15–30 个 forward ACTIONABLE 后】

**从 5A 升级到 5B 的条件：**
- [ ] 新增 15–30 个 forward ACTIONABLE_CANDIDATE
- [ ] Forward ACTIONABLE avg_alpha > 0
- [ ] Forward ACTIONABLE avg_R > 0.25R
- [ ] ACTIONABLE 表现优于 WATCHLIST
- [ ] worst loss 可控
- [ ] 执行纪律稳定
- [ ] 结果不是由 1–2 个 ticker 贡献

**仓位规则：** 每笔账户风险 ≤ 0.25%–0.5%，每次最多 1–2 笔，避免同 sector / same catalyst 过度集中

### Phase 5C — 正式策略化【3–6 个月后评估】

这不是开始小仓的时间点，而是判断是否可以更正式依赖系统 / 放大仓位的阶段。

**前置条件：**
- [ ] 50+ forward signals
- [ ] 30+ forward ACTIONABLE_CANDIDATE
- [ ] 不同 market regime 下仍保持 positive alpha
- [ ] score ≥52 rolling alpha 持续 > 0
- [ ] ACTIONABLE avg_R 明显高于 WATCHLIST
- [ ] position-size-adjusted drawdown 可控
- [ ] 不是少数 ticker / sector 贡献大部分收益

**仍然不做：** 全自动交易 / 大仓位 / 无人工确认交易

---

## 当前明确不做

| 方向 | 原因 |
|------|------|
| 全自动交易 | forward validation 未完成 |
| 直接用 score 越高越好排序 | 高分段不单调，55+ 反而变弱 |
| 恢复 77 ACTIONABLE 阈值 | 历史上不可达 |
| 降 WATCHLIST 到 50 | 50–52 是边缘区，混有噪音 |
| macro_watch / sympathy_play / opinion_watch | 历史表现接近噪音 |
| low-volume earnings 自动加分 | false upgrade 来源之一 |
| weak-confirmation gap-down SHORT | 容易追跌失败 |
| 纯价格层信号 | 缺乏独立 edge |
| 无人工确认实盘 | 当前系统定位仍是研究助手 |
| 为单个 missed winner 调规则 | 容易 overfit |

---

## 当前系统定位

Alpha Engine 是**事件驱动研究助手**，不是自动交易系统。

**系统职责：** 过滤噪音 / 标记 WATCHLIST / ACTIONABLE_CANDIDATE / 给出 score decomposition / 给出 risk / position size 建议 / 记录 forward outcome / 帮助人更快发现值得研究的机会

**人的职责：** 判断事件质量 / 判断 catalyst 是否真实 / 判断市场环境是否适合 / 判断是否存在流动性 / gap / execution 风险 / 决定是否 paper trade / 极小仓 live experiment / 小仓验证

---

## 当前最重要的 Next Actions

1. [ ] 上线新 tier，开始 forward tracking
2. [ ] 把 ACTIONABLE 理解为 ACTIONABLE_CANDIDATE / HIGH_PRIORITY
3. [ ] 增加 NEAR_WATCHLIST reason 显示
4. [ ] 做 position-size-adjusted portfolio backtest
5. [x] 完整记录 earn_strength（已完成）
6. [ ] 每周比较 ACTIONABLE vs WATCHLIST vs IGNORE 的 alpha / R
7. [ ] 可以开始极小仓 live experiment，每笔风险控制在 0.1%–0.25%
8. [ ] 累积 15–30 个 forward ACTIONABLE 后，再决定是否升级到小仓验证
