"""
db.py — SQLite schema + helpers
"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "alpha.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS news_articles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT,
    title            TEXT,
    url              TEXT UNIQUE,
    published_at     TEXT,
    content          TEXT,
    fetched_at       TEXT DEFAULT (datetime('now')),
    symbols          TEXT,          -- JSON list e.g. '["NVDA","AMD"]'
    event_type       TEXT,          -- earnings / macro / product / layoff / ma / regulation / ai
    sentiment_score  REAL,          -- -1.0 to 1.0
    novelty_score    REAL,          -- 0-1, is this new info?
    importance_score REAL           -- 0-1
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    snapshot_date TEXT,
    close_price   REAL,
    change_pct    REAL,
    volume        REAL,
    avg_volume    REAL,
    volume_ratio  REAL,             -- volume / avg_volume
    rsi_14        REAL,
    ma_20         REAL,
    ma_50         REAL,
    above_ma20    INTEGER,          -- 0 or 1
    week_high_52  REAL,
    week_low_52   REAL,
    market_cap    REAL,
    atr_14        REAL,             -- 14-day Average True Range
    UNIQUE(symbol, snapshot_date)
);

CREATE TABLE IF NOT EXISTS trade_candidates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date           TEXT,
    symbol             TEXT,
    company_name       TEXT,
    direction          TEXT,           -- LONG / SHORT
    final_score        REAL,           -- 0-100
    event_score        REAL,
    sentiment_score    REAL,
    technical_score    REAL,
    thesis             TEXT,           -- 1-2 sentence core reason
    entry_note         TEXT,           -- specific price/condition
    stop_loss_note     TEXT,
    target_note        TEXT,
    risk_note          TEXT,           -- what would invalidate this
    action             TEXT,           -- ACTIONABLE / WATCHLIST / MONITOR / IGNORE
    news_ids           TEXT,           -- JSON list of supporting news ids
    event_edge_score   REAL,           -- Layer 2A: 0-25
    market_conf_score  REAL,           -- Layer 2B: 0-20
    regime_fit_score   REAL,           -- Layer 2C: 0-15
    relative_opp_score REAL,           -- Layer 2D: 0-15
    freshness_score    REAL,           -- Layer 2E: 0-10
    risk_penalty_score REAL,           -- Layer 2F: 0-15 (subtracted)
    strategy_bucket    TEXT,           -- event_long/event_short/sympathy_play/technical_follow
    UNIQUE(run_date, symbol)
);

CREATE TABLE IF NOT EXISTS watched_symbols (
    symbol   TEXT PRIMARY KEY,
    sector   TEXT,
    priority INTEGER DEFAULT 1,     -- 1=high, 2=medium, 3=low
    enabled  INTEGER DEFAULT 1,
    note     TEXT
);

CREATE TABLE IF NOT EXISTS daily_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date    TEXT UNIQUE,
    run_at      TEXT,
    news_fetched INTEGER,
    prices_fetched INTEGER,
    candidates_found INTEGER,
    market_regime TEXT,             -- bull / bear / neutral / choppy
    spy_change_pct REAL,
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS signal_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    signal_date     TEXT,           -- date the signal was generated
    signal          TEXT,           -- ACTIONABLE / WATCHLIST / MONITOR
    final_score     REAL,
    regime          TEXT,           -- bull / bear / neutral / choppy
    direction       TEXT,           -- LONG / SHORT
    strategy_bucket TEXT,
    entry_price     REAL,           -- close price on signal_date
    t3_date         TEXT,           -- calendar of t+3 trading day
    t5_date         TEXT,
    t10_date        TEXT,
    t3_price        REAL,
    t5_price        REAL,
    t10_price       REAL,
    t3_pnl_pct      REAL,           -- signed % from entry (LONG positive = up)
    t5_pnl_pct      REAL,
    t10_pnl_pct     REAL,
    outcome         TEXT DEFAULT 'PENDING',   -- PENDING / WIN / LOSS / SCRATCH
    notes           TEXT,
    UNIQUE(symbol, signal_date)
);
"""

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def migrate_db():
    """Add new columns to existing DBs without breaking older schemas."""
    conn = get_conn()
    migrations = [
        ("price_snapshots",  "atr_14",             "REAL"),
        ("price_snapshots",  "day_high",           "REAL"),
        ("price_snapshots",  "day_low",            "REAL"),
        ("trade_candidates", "event_edge_score",   "REAL"),
        ("trade_candidates", "market_conf_score",  "REAL"),
        ("trade_candidates", "regime_fit_score",   "REAL"),
        ("trade_candidates", "relative_opp_score", "REAL"),
        ("trade_candidates", "freshness_score",    "REAL"),
        ("trade_candidates", "risk_penalty_score", "REAL"),
        ("trade_candidates", "strategy_bucket",    "TEXT"),
        # signal_outcomes extended columns
        ("signal_outcomes",  "notes",           "TEXT"),
        ("signal_outcomes",  "t1_date",         "TEXT"),
        ("signal_outcomes",  "t1_price",        "REAL"),
        ("signal_outcomes",  "t1_pnl_pct",      "REAL"),
        ("signal_outcomes",  "stop_price",      "REAL"),
        ("signal_outcomes",  "target_price",    "REAL"),
        ("signal_outcomes",  "atr_at_signal",   "REAL"),
        ("signal_outcomes",  "paper_pnl_pct",   "REAL"),
        ("signal_outcomes",  "paper_exit",      "TEXT"),   # HIT_STOP/HIT_TARGET/T5_EXIT/PENDING
        ("signal_outcomes",  "event_type",      "TEXT"),   # earnings/macro/ma/ai/product/regulation/layoff/general
        ("daily_runs",       "steps_json",      "TEXT"),   # JSON array of per-step {step,ok,ms,detail}
        ("trade_candidates", "thesis_conviction", "TEXT"),
        ("trade_candidates", "thesis_technical",  "TEXT"),
        ("trade_candidates", "thesis_news",       "TEXT"),
        ("trade_candidates", "thesis_risk",       "TEXT"),
    ]
    for table, col, coltype in migrations:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()

    # Seed default watchlist — S&P 500 core + high-vol names
    defaults = [
        ("SPY",  "ETF",           1), ("QQQ",  "ETF",           1),
        ("NVDA", "Technology",    1), ("AMD",  "Technology",    1),
        ("MSFT", "Technology",    1), ("AAPL", "Technology",    1),
        ("GOOGL","Technology",    1), ("META", "Technology",    1),
        ("AMZN", "Technology",    1), ("TSLA", "Automotive",    1),
        ("JPM",  "Finance",       1), ("BAC",  "Finance",       2),
        ("GS",   "Finance",       2), ("PLTR", "Technology",    1),
        ("CRM",  "Technology",    2), ("NFLX", "Media",         2),
        ("UBER", "Technology",    2), ("COIN", "Finance",       2),
        ("SOFI", "Finance",       2), ("ARM",  "Technology",    1),
        ("SMCI", "Technology",    2), ("MU",   "Technology",    2),
        ("AVGO", "Technology",    1), ("TSM",  "Technology",    1),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO watched_symbols (symbol, sector, priority) VALUES (?,?,?)",
        defaults
    )
    conn.commit()
    conn.close()
    migrate_db()
    print(f"[db] initialized → {DB_PATH}")

if __name__ == "__main__":
    init_db()
