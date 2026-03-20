# Alpha Engine — Evolution Plan

> 目标：从"每天扫描 31 只股票的打分工具"进化为"可以指导投资决策的研究助手"
>
> 当前状态：数据管道稳定，信号质量未经验证（胜率 35%，40 个样本）
> 核心原则：少而精 > 多而杂；先验证再扩展；人工判断是最终门

---

## Phase 0 — 已完成的基础工作

- [x] 数据管道：新闻采集 → 价格 → 分析 → 报告 → 通知
- [x] 4 层评分：EventEdge + MarketConf + RegimeFit + RelOpp - RiskPenalty
- [x] 信号追踪：t1/t3/t5 结果记录，paper trade 止损/目标
- [x] Evaluation tab：signal stability、score buckets、paper trade summary
- [x] 三条保护规则：RSI 超卖过滤 + EventEdge 分层 cap + 相似度 crowding penalty
- [x] Backtest：价格层单独无 edge，确认 EventEdge 是主要 alpha 来源

---

## Phase 1 — 让系统变得"少而精" 【当前阶段】

**目标**：把每日输出从 ~30 条噪音缩减为 2-3 条值得认真对待的候选

### 1.1 加 Eligibility Gate（硬门槛过滤）

必须全部满足才进入候选池：

- [ ] EventEdge ≥ 15（有真实事件驱动，非纯价格信号）
- [ ] 方向 = LONG（SHORT 在现有数据未验证，暂禁）
- [ ] 制度 ≠ bear（bear 中不开新 LONG）
- [ ] catalyst ∈ {earnings, product, regulation, ma}（排除 general/macro 纯叙事）
- [ ] 非严重拥挤（同日同方向同 sector ≤ 1 条进入候选）

### 1.2 每日输出上限

- [ ] 全局：每天最多 3 条 actionable（超出按 retention_priority 截断）
- [ ] 同 sector：最多 1 条高优先级
- [ ] 同方向：最多 2 条

### 1.3 改进输出格式

每条候选输出以下字段（现在 thesis 缺部分）：

- [ ] 事件驱动（event thesis）
- [ ] 为什么现在（why now）
- [ ] 为什么不拥挤（why not crowded）
- [ ] 失效条件（invalidation：什么情况说明 thesis 错了）
- [ ] 对比基准（vs SPY / sector ETF）

---

## Phase 2 — 专注一个 Setup，积累验证样本【1-2 个月】

**目标**：用 post-earnings drift LONG 积累 30+ 个真实样本，验证是否有 edge

### 2.1 专注 post_earnings_drift

- [ ] 过滤规则：bucket = post_earnings_drift + LONG + EE ≥ 15
- [ ] 每次财报后系统自动识别并标记
- [ ] 纸交易记录要严格：信号生成时固定，不能事后修改

### 2.2 加 Excess Return 追踪

比绝对涨跌更重要的是超额收益——是否跑赢市场 beta？

- [ ] DB 新增字段：`spy_return_t5`、`sector_return_t5`
- [ ] 计算：`excess_return = t5_pnl - spy_return_t5`
- [ ] Evaluation tab 展示：raw return vs excess return

### 2.3 改主 KPI

- [ ] 从"胜率"改为"expectancy = 胜率 × 平均盈 - 败率 × 平均亏"
- [ ] 追踪 profit factor = 总盈利 / 总亏损
- [ ] 追踪 top bucket 相对 lower bucket 的单调性（验证评分有效性）

---

## Phase 3 — 重新校准评分权重【3 个月后，≥80 个已解决样本】

**目标**：让 score 真正预测收益排序，而不只是叙事完整度

### 3.1 手动调权（等不及 optimizer 时的临时方案）

基于已知 backtest 结论：

- [ ] RegimeFit 权重下调（当前 r=-0.236，负向预测）
- [ ] MarketConf 权重上调（唯一正向 r=+0.237）

### 3.2 数据驱动校准

- [ ] 按 {EE tier × catalyst × regime × direction} 分桶，估计每桶 expected excess return
- [ ] 运行 `weight_optimizer`（需 ≥80 个已解决样本）
- [ ] Walk-forward 验证：train 60% / validate 20% / forward 20%

### 3.3 把 score 从"叙事分"改成"预期收益排序"

- [ ] 用分桶 expected return 替换当前加权求和逻辑（或作为调整层叠加）

---

## Phase 4 — 实盘小仓位验证【6 个月后，胜率 >55% 且样本 >30】

**前置条件（全部满足才考虑）：**

- [ ] post_earnings_drift LONG 在 30+ 样本上胜率 > 55%
- [ ] excess return（超额）为正
- [ ] walk-forward 验证通过（不是 in-sample 结果）
- [ ] 最大单日回撤可控（< 3%）

**仓位原则：**

- 单笔风险 ≤ 0.5%（更保守，低于当前 paper 的 0.75%）
- 每次最多 1 个持仓
- 严格按系统止损，不主观持有

---

## 长期不做的事（当前阶段明确排除）

| 方向 | 原因 |
|---|---|
| Broad market regime timing | 当前最弱，无验证 |
| Neutral/choppy 下 SHORT | 数据显示 SHORT 整体胜率 12% |
| 纯价格层信号（EE < 8）| Backtest 确认无独立 edge |
| 提前抓反转 | 系统无此能力 |
| 全自动交易（无人工确认） | 数据不足以支撑 |

---

## 当前关键指标（更新于 2026-03-20）

| 指标 | 数值 | 目标 |
|---|---|---|
| 已解决信号数 | 40 | ≥ 80 才能校准权重 |
| 整体胜率（t5） | 35% | — |
| LONG 胜率 | 69% | 验证 post_earnings_drift 子集 |
| SHORT 胜率 | 12% | 当前阶段不追这个 |
| 平均 t5 pnl | -0.03% | 需转正 |
| Excess return vs SPY | 未追踪 | Phase 2 加入 |
| post_earnings_drift 样本数 | ~5 | 目标 30+ |

---

## 系统当前定位

> **研究助手，不是自动交易系统。**
>
> 系统的职责：每天筛选出 1-3 条经过严格门槛的候选，并说明为什么值得看、最大风险是什么。
> 人的职责：做最终判断，决定是否纸交易/实盘。
