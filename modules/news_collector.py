"""
news_collector.py — pulls from Yahoo Finance RSS + NewsAPI (free tier)
Stores raw articles into news_articles table.
"""
import feedparser, requests, json, re
from datetime import datetime, timezone, timedelta
from typing import Optional
from modules.db import get_conn


MAX_ARTICLE_AGE_HOURS = 36  # drop anything older than this


def parse_pub_date(entry) -> Optional[str]:
    """
    Return a UTC ISO-8601 string from a feedparser entry, or None if unparseable.
    feedparser always gives us `published_parsed` as a UTC time.time struct when
    the feed includes a valid date — use that instead of the raw string.
    """
    tp = getattr(entry, "published_parsed", None)
    if tp:
        try:
            dt = datetime(*tp[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass
    # fallback: try raw string via email.utils
    raw = getattr(entry, "published", None)
    if raw:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw).astimezone(timezone.utc)
            return dt.isoformat()
        except Exception:
            pass
    return None

# ── RSS feeds (free, no API key) ──────────────────────────────────────────
RSS_FEEDS = [
    ("Yahoo Finance",    "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch",      "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
    ("Seeking Alpha",    "https://seekingalpha.com/market_currents.xml"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC Markets",     "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("Benzinga",         "https://www.benzinga.com/feed"),
]

# ── Ticker extractor ─────────────────────────────────────────────────────
TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

SP500_SAMPLE = {
    "NVDA","AMD","MSFT","AAPL","GOOGL","META","AMZN","TSLA","JPM","BAC",
    "GS","PLTR","CRM","NFLX","UBER","COIN","SOFI","ARM","SMCI","MU",
    "AVGO","TSM","SPY","QQQ","INTC","QCOM","TXN","ORCL","ADBE","SNOW",
    "SHOP","SQ","PYPL","V","MA","BRK","JNJ","UNH","PFE","LLY",
    "XOM","CVX","HAL","NKE","DIS","WMT","COST","HD","MCD","SBUX",
}

# Company name → ticker mappings (handles how journalists write names)
NAME_TO_TICKER = {
    "nvidia": "NVDA", "nvda": "NVDA",
    "advanced micro": "AMD", "amd": "AMD",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "alphabet": "GOOGL", "google": "GOOGL",
    "meta platforms": "META", "meta ai": "META", "meta's": "META",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS", "goldman": "GS",
    "palantir": "PLTR",
    "salesforce": "CRM",
    "netflix": "NFLX",
    "coinbase": "COIN",
    "sofi": "SOFI",
    "broadcom": "AVGO",
    "taiwan semiconductor": "TSM", "tsmc": "TSM",
    "micron": "MU",
    "intel": "INTC",
    "snowflake": "SNOW",
    "shopify": "SHOP",
    "paypal": "PYPL",
    "oracle": "ORCL",
    "adobe": "ADBE",
    "qualcomm": "QCOM",
    "uber": "UBER",
    "arm holdings": "ARM", "arm's": "ARM",
    "supermicro": "SMCI",
    "bytedance": "NVDA",  # ByteDance stories usually move NVDA (chip export)
}

def extract_tickers(text: str) -> list:
    text_lower = (text or "").lower()
    found = set()

    # 1. Direct ticker symbols (uppercase words)
    for t in TICKER_RE.findall(text or ""):
        if t in SP500_SAMPLE:
            found.add(t)

    # 2. Company name lookup
    for name, ticker in NAME_TO_TICKER.items():
        if name in text_lower:
            found.add(ticker)

    return list(found)

# ── Event classifier (keyword-based, fast, no API cost) ──────────────────
EVENT_KEYWORDS = {
    "earnings":   ["earnings", "eps", "revenue", "quarterly", "beat", "miss", "guidance", "profit"],
    "macro":      ["fed", "fomc", "rate", "inflation", "cpi", "gdp", "jobs", "unemployment", "recession"],
    "product":    ["launch", "release", "announce", "unveil", "debut", "product", "model"],
    "layoff":     ["layoff", "layoffs", "job cuts", "workforce reduction", "restructur", "downsiz"],
    "ma":         ["acqui", "merger", "deal", "takeover", "buyout", "bid for"],
    "regulation": ["sec", "ftc", "doj", "regulation", "antitrust", "ban", "lawsuit", "fine"],
    "ai":         ["artificial intelligence", "ai model", "llm", "gpt", "gemini", "claude", "openai", "deepseek"],
}

def classify_event(text):
    text_lower = (text or "").lower()
    for event, keywords in EVENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return event
    return "general"

# ── Sentiment (keyword-based, fast, no API cost) ──────────────────────────
POSITIVE = ["beat", "surge", "rally", "strong", "record", "growth", "bullish",
            "upgrade", "outperform", "buy", "raise", "positive", "profit", "gain"]
NEGATIVE = ["miss", "drop", "fall", "weak", "cut", "downgrade", "underperform",
            "sell", "loss", "decline", "concern", "risk", "warn", "layoff", "crash"]

def quick_sentiment(text):
    t = (text or "").lower()
    pos = sum(1 for w in POSITIVE if w in t)
    neg = sum(1 for w in NEGATIVE if w in t)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)

# ── Novelty (simple: have we seen this title before?) ─────────────────────
def novelty_score(title, conn):
    row = conn.execute(
        "SELECT COUNT(*) as c FROM news_articles WHERE title LIKE ?",
        (f"%{title[:40]}%",)
    ).fetchone()
    return 0.2 if row["c"] > 0 else 1.0

# ── Importance heuristics ──────────────────────────────────────────────────
IMPORTANCE_BOOST = {
    "earnings": 0.9, "macro": 0.85, "ma": 0.8,
    "ai": 0.75, "layoff": 0.65, "regulation": 0.7, "product": 0.6, "general": 0.3
}

def importance_score(event_type, tickers, sentiment_abs):
    base = IMPORTANCE_BOOST.get(event_type, 0.3)
    ticker_boost = min(len(tickers) * 0.05, 0.2)
    return round(min(base + ticker_boost + sentiment_abs * 0.1, 1.0), 2)

# ── Main collector ─────────────────────────────────────────────────────────
def collect_news(verbose=True):
    conn = get_conn()
    saved = 0
    skipped = 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)

    for source_name, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:  # last 30 per feed
                title   = getattr(entry, "title", "")
                link    = getattr(entry, "link", "")
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")

                if not title or not link:
                    continue

                pub = parse_pub_date(entry)
                if pub is None:
                    # no parseable date — assume it's fresh (don't discard)
                    pub = datetime.now(timezone.utc).isoformat()
                else:
                    # drop stale articles
                    try:
                        pub_dt = datetime.fromisoformat(pub)
                        if pub_dt < cutoff:
                            skipped += 1
                            continue
                    except Exception:
                        pass

                full_text = f"{title} {summary}"
                tickers   = extract_tickers(full_text)
                event     = classify_event(full_text)
                sent      = quick_sentiment(full_text)
                nov       = novelty_score(title, conn)
                imp       = importance_score(event, tickers, abs(sent))

                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO news_articles
                        (source, title, url, published_at, content, symbols,
                         event_type, sentiment_score, novelty_score, importance_score)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (source_name, title, link, pub, summary,
                          json.dumps(tickers), event, sent, nov, imp))
                    saved += 1
                except Exception:
                    skipped += 1

        except Exception as e:
            if verbose:
                print(f"[news] ⚠ {source_name}: {e}")

    conn.commit()
    conn.close()
    if verbose:
        print(f"[news] collected {saved} articles, skipped {skipped} duplicates")
    return saved
