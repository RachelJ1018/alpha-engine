# Alpha Engine

A personal, local investment research tool for swing trading. Runs daily to provide systematic, research-driven trading signals with explainable scoring and risk-adjusted position sizing.

**Core Mission:** Replace reactive market-watching with daily automated analysis that tells you: **operate / don't operate / how to operate**.

---

## Features

### News Aggregation
- Fetches from 6 RSS sources: Yahoo Finance, MarketWatch, Reuters, CNBC, Seeking Alpha, Benzinga
- Automatic ticker extraction via symbol regex and company name mapping
- Event classification: earnings, macro, product, M&A, regulation, AI, layoffs, general
- Sentiment scoring (-1.0 to +1.0) using positive/negative keyword analysis
- Novelty detection to filter duplicate stories

### Technical Analysis
- 60-day daily OHLCV data via yfinance
- Technical indicators: RSI(14), MA(20/50), ATR(14), volume ratio
- Market regime detection (SPY-based): bull / bear / neutral / choppy

### Multi-Layer Scoring System
Six independent scoring layers (0-100 total):

| Layer | Points | Description |
|-------|--------|-------------|
| Event Edge | 0-25 | Event importance + source credibility + sentiment |
| Market Confidence | 0-20 | Technical setup quality (RSI, MA, volume) |
| Regime Fit | 0-15 | Alignment with current market environment |
| Relative Opportunity | 0-15 | Risk/reward potential + gap dynamics |
| Freshness | 0-10 | News recency bonus |
| Risk Penalty | 0-15 | Deducted for volatility & adverse technicals |

**Action Labels:**
- **ACTIONABLE** (score >= 70 bull / 75 neutral): Ready-to-trade setups
- **WATCHLIST** (score 55-69): Monitor for entry clarity
- **MONITOR** (score 40-54): Track for next opportunity
- **IGNORE** (score < 40): Poor risk/reward

### AI-Generated Trade Theses
- Claude API generates specific entry/stop/target guidance
- Includes: core reason, entry condition, stop-loss rationale, target calculation, invalidation scenario
- Cost: ~$0.01-0.05 per run

### Risk Engine
- **ATR-based stops:** Stop = entry +/- 1.2x 14-day ATR
- **2R targets:** Target = entry +/- 2.0x stop distance
- **Regime scaling:** Bull (1.0x) -> Neutral (0.8x) -> Bear (0.7x) -> Choppy (0.5x)
- **Portfolio constraints:** Max positions, single-name exposure limits, sector concentration

### Report Generation
- Dual output: HTML (browser) + Markdown (text)
- Market regime banner with SPY context
- Per-candidate breakdown: price, technicals, layered scores, thesis, entry/stop/target
- Key news articles from last 24h

### Backtesting
- Earnings-driven backtest (5-year history)
- News-event backtest with outcome tracking
- Signal outcome tracking at t+1, t+3, t+5, t+10 days

### Interactive Dashboard
- Streamlit-based UI with portfolio configuration
- Real-time pipeline execution
- Score breakdown visualization
- Historical signal outcomes

---

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Claude API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add to `~/.zshrc` or `~/.bashrc` for persistence.

### 3. Initialize

```bash
python run_daily.py
```

Creates `data/alpha.db` and opens today's report.

---

## Usage

### Daily CLI

```bash
# Full pipeline (news + prices + analysis + report)
python run_daily.py

# News fetching only
python run_daily.py --news-only

# Price fetching only
python run_daily.py --price-only

# Regenerate report from cached data
python run_daily.py --report

# Silent run (no browser)
python run_daily.py --no-open
```

### Interactive Dashboard

```bash
streamlit run app.py
```

### Backtesting

```bash
# Earnings-driven backtest
python -m modules.backtest --tickers NVDA TSLA --years 3

# News-event backtest
python -m modules.news_event_backtest --query
```

### Database Access

```bash
sqlite3 data/alpha.db
```

---

## Automation (cron)

Run at 7:30am Mon-Fri before market open:

```bash
crontab -e
```

Add:
```
30 7 * * 1-5 cd /path/to/alphalocal && /usr/bin/python3 run_daily.py --no-open >> /tmp/alpha.log 2>&1
```

---

## Watchlist Management

### Via SQLite

```sql
-- Add a symbol
INSERT INTO watched_symbols (symbol, sector, priority) VALUES ('HOOD', 'Finance', 1);

-- Disable a symbol
UPDATE watched_symbols SET enabled=0 WHERE symbol='BAC';

-- View current watchlist
SELECT * FROM watched_symbols WHERE enabled=1 ORDER BY priority;
```

### Via Code

Edit the `defaults` list in `modules/db.py` and re-run.

---

## Project Structure

```
alpha-engine/
├── run_daily.py              # Main entry point
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── db.py                 # SQLite schema & helpers
│   ├── news_collector.py     # RSS aggregation & scoring
│   ├── price_fetcher.py      # yfinance + technicals
│   ├── analyzer.py           # Multi-layer scoring + Claude thesis
│   ├── report_generator.py   # HTML/markdown output
│   ├── risk_engine.py        # Position sizing & risk management
│   ├── signal_tracker.py     # Historical outcome tracking
│   ├── backtest.py           # Earnings backtester
│   └── news_event_backtest.py # Event-driven backtester
│
├── data/
│   ├── alpha.db              # SQLite database (auto-created)
│   ├── backtest_results.json
│   └── news_event_backtest_results.json
│
└── reports/
    ├── report_YYYY-MM-DD.html
    ├── report_YYYY-MM-DD.md
    ├── latest.md
    └── backtest_*.html
```

---

## Database Schema

| Table | Purpose |
|-------|---------|
| `news_articles` | Source, title, URL, symbols, event type, sentiment, novelty, importance |
| `price_snapshots` | OHLCV, RSI, MA20/50, volume ratio, ATR, 52-week range, market cap |
| `trade_candidates` | Scoring layers, thesis, entry/stop/target, action labels |
| `watched_symbols` | Ticker universe with priority/sector flags |
| `daily_runs` | Aggregate metrics (news count, candidates, regime, SPY change) |
| `signal_outcomes` | Historical signals with t+1/3/5/10 P&L tracking |

---

## Report Output Example

For each candidate:

| Field | Example |
|-------|---------|
| **Action** | BUY NOW / WATCH / MONITOR |
| **Score** | 78/100 |
| **What & Why** | "NVDA beat earnings by 18% with strong data center guidance." |
| **Entry** | "Above $487 on volume > 1.5x avg" |
| **Stop Loss** | "Below $472 — breaks 20-day MA" |
| **Target** | "$510 in 3-5 days (prior resistance)" |
| **Invalidation** | "If QQQ breaks below 200-day MA, thesis is off." |

---

## Risk Rules

- Max position size: **2% of portfolio**
- Hard stop loss: **-2% from entry**
- Take profit target: **+5%**, trail after +3%
- Max open positions: **5**
- If SPY drops >1.5% intraday: **go to cash**

---

## Daily Workflow

1. **Trigger** (7:30am via cron or manual)
2. **News Collection** - RSS feeds with ticker extraction + event/sentiment classification
3. **Price Fetch** - yfinance for technicals (RSI, MA, ATR, volume ratio)
4. **Market Regime** - SPY analysis for regime classification
5. **Candidate Scoring** - 6-layer algorithm with Claude thesis generation
6. **Risk Engine** - Position sizing: shares, stop, target, P&L expectations
7. **Report Generation** - HTML + markdown outputs
8. **Signal Recording** - Log ACTIONABLE/WATCHLIST/MONITOR candidates
9. **Outcome Tracking** - Update t+1/3/5/10 P&L on prior signals
10. **Browser Open** - Display report (unless --no-open)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Data Fetching | yfinance, feedparser |
| Database | SQLite3 |
| Analysis | pandas, numpy |
| UI | Streamlit |
| AI | Anthropic Claude API |
| Reports | HTML, Markdown |
| Automation | Python CLI, cron |

---

## Cost

- **News + prices:** Free (RSS feeds + yfinance)
- **Claude API:** ~$0.01-0.05 per daily run (Sonnet, 20-30 tickers)
- **Total:** Essentially free for personal use

---

## Future Enhancements

- **Options data** - Add Polygon.io API integration
- **Social sentiment** - Reddit/Twitter analysis
- **Broker integration** - robin_stocks for portfolio sync
- **Email alerts** - SMTP notifications
- **Extended backtesting** - Historical event comparison
