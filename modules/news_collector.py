"""
news_collector.py — pulls from Yahoo Finance RSS + NewsAPI (free tier)
Stores raw articles into news_articles table.
"""
import feedparser, requests, json, re, os
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

# Institutions that often appear as *analyst* ("Goldman upgrades X") rather
# than as the stock being written about.  Used in extract_tickers() below.
_ANALYST_NAMES = frozenset({
    "bank of america", "goldman sachs", "goldman",
    "jpmorgan", "jp morgan", "morgan stanley", "wells fargo",
    "barclays", "ubs", "mizuho", "needham", "jefferies",
    "piper sandler", "raymond james", "deutsche bank",
    "bernstein", "wedbush", "keybanc", "stifel",
})

# Verb phrases that, when found within ~80 chars AFTER an institution name,
# signal it is acting as analyst-of-another-stock, not as the news subject.
_ANALYST_VERBS = (
    " is betting", " bets ", " upgrade", " downgrade",
    " initiat", " reiter",
    " maintains a buy", " maintains a sell", " maintains hold",
    " price target on", " target on ", " target price",
    " bullish on ", " bearish on ",
    " rates ", " cut price", " raise price",
)


def _is_analyst_action(name: str, text_lower: str) -> bool:
    """True if the institution name is followed by an analyst-action verb."""
    idx = text_lower.find(name)
    if idx == -1:
        return False
    snippet = text_lower[idx: idx + len(name) + 80]
    return any(v in snippet for v in _ANALYST_VERBS)


def extract_tickers(text: str, title: str = "") -> list:
    """
    Extract watchlist tickers from article text.

    title (optional): when provided, enables two quality improvements:
      1. Analyst-context filter: if a financial institution name
         (e.g. "Bank of America") appears in analyst-action context
         ("is betting that QCOM will underperform") AND other tickers
         are present, the institution ticker is dropped — it's the
         analyst, not the subject.
      2. Title-priority cap: tickers found only in the body (not the
         headline) are capped at 2 when title has matches, or dropped
         entirely when body has >3 body-only tickers and title has none
         (prevents TipRanks/PR wire sidebar lists from inflating scores).
    """
    text_lower  = (text  or "").lower()
    title_lower = (title or "").lower()

    # ── Step 1: collect all raw candidates from full text ─────────────────
    raw: set = set()
    for t in TICKER_RE.findall(text or ""):
        if t in SP500_SAMPLE:
            raw.add(t)
    for name, ticker in NAME_TO_TICKER.items():
        if name in text_lower:
            raw.add(ticker)

    # ── Step 2: analyst-context filter (needs title for best accuracy) ────
    found: set = set()
    for name, ticker in NAME_TO_TICKER.items():
        if name not in text_lower:
            continue
        others = raw - {ticker}
        # Drop institution ticker when: it's a known analyst name,
        # other stocks are present (the actual subjects), and it appears
        # right before an analyst-action verb.
        if (name in _ANALYST_NAMES
                and others
                and _is_analyst_action(name, text_lower)):
            continue
        found.add(ticker)

    # Re-add direct TICKER_RE matches — they bypass the name filter
    for t in TICKER_RE.findall(text or ""):
        if t in SP500_SAMPLE:
            found.add(t)

    if not title:
        return list(found)

    # ── Step 3: title-priority cap (blocks sidebar ticker lists) ──────────
    title_found: set = set()
    for t in TICKER_RE.findall(title):
        if t in SP500_SAMPLE:
            title_found.add(t)
    for name, ticker in NAME_TO_TICKER.items():
        if name not in title_lower:
            continue
        others = found - {ticker}
        if (name in _ANALYST_NAMES
                and others
                and _is_analyst_action(name, title_lower)):
            continue
        title_found.add(ticker)

    body_only = found - title_found
    if title_found:
        # A few extra body mentions are fine
        body_cap = 2
    elif len(body_only) <= 3:
        # No title tickers, small body set — probably legitimate
        body_cap = len(body_only)
    else:
        # No title tickers + large body set = sidebar/related-ticker dump
        body_cap = 0

    return list(title_found | set(sorted(body_only)[:body_cap]))

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

# ── Helpers for API-based sources ────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Pull a clean source name from a URL: 'https://www.reuters.com/...' → 'Reuters'.
    This feeds into SOURCE_TIER in analyzer.py via lowercase substring matching."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.replace("www.", "")
        return host.split(".")[0].title()
    except Exception:
        return "Web"


def _parse_relative_date(date_str: str) -> str:
    """Convert SerpAPI relative dates ('3 hours ago', '1 day ago') to ISO-8601 UTC."""
    now = datetime.now(timezone.utc)
    if not date_str:
        return now.isoformat()
    s = date_str.lower().strip()
    try:
        m = re.search(r'(\d+)', s)
        n = int(m.group(1)) if m else 1
        if "minute" in s: return (now - timedelta(minutes=n)).isoformat()
        if "hour"   in s: return (now - timedelta(hours=n)).isoformat()
        if "day"    in s: return (now - timedelta(days=n)).isoformat()
        if "week"   in s: return (now - timedelta(weeks=n)).isoformat()
        return datetime.fromisoformat(date_str).isoformat()
    except Exception:
        return now.isoformat()


def _store_article(conn, source, title, url, pub, content, tickers, event, sent, nov, imp):
    """INSERT OR IGNORE a single article. Returns True if new."""
    try:
        conn.execute("""
            INSERT OR IGNORE INTO news_articles
            (source, title, url, published_at, content, symbols,
             event_type, sentiment_score, novelty_score, importance_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (source, title, url, pub, content[:2000],
              json.dumps(tickers), event, sent, nov, imp))
        return True
    except Exception:
        return False


# ── Tavily Search ─────────────────────────────────────────────────────────

def collect_tavily_news(symbols: list, api_key: str, days: int = 2,
                        verbose: bool = True) -> int:
    """
    Search Tavily for each symbol and store results.
    Free tier: 1,000 requests/month — at 15 symbols/day that's ~450/month.
    Source name is extracted from the article URL (Reuters, Bloomberg, etc.)
    so it flows into SOURCE_TIER scoring in analyzer.py automatically.
    """
    conn = get_conn()
    saved = 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)

    for sym in symbols:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":      api_key,
                    "query":        f"{sym} stock news",
                    "topic":        "news",
                    "days":         days,
                    "max_results":  5,
                    "search_depth": "basic",
                },
                timeout=15,
            )
            if r.status_code != 200:
                if verbose:
                    print(f"[tavily] {sym}: HTTP {r.status_code} — {r.text[:80]}")
                continue

            for item in r.json().get("results", []):
                title   = (item.get("title") or "").strip()
                url     = item.get("url", "")
                content = item.get("content", "")
                pub_raw = item.get("published_date", "")

                if not title or not url:
                    continue

                pub = datetime.now(timezone.utc).isoformat()
                if pub_raw:
                    try:
                        pub = datetime.fromisoformat(
                            pub_raw.replace("Z", "+00:00")
                        ).isoformat()
                    except Exception:
                        pass

                try:
                    if datetime.fromisoformat(pub) < cutoff:
                        continue
                except Exception:
                    pass

                source_name = _extract_domain(url)
                full_text   = f"{title} {content}"
                tickers     = extract_tickers(full_text, title=title)
                if sym not in tickers:
                    tickers.append(sym)

                event = classify_event(full_text)
                sent  = quick_sentiment(full_text)
                nov   = novelty_score(title, conn)
                imp   = importance_score(event, tickers, abs(sent))

                if _store_article(conn, source_name, title, url, pub,
                                  content, tickers, event, sent, nov, imp):
                    saved += 1

        except Exception as e:
            if verbose:
                print(f"[tavily] {sym}: {e}")

    conn.commit()
    conn.close()
    return saved


# ── SerpAPI Google News ───────────────────────────────────────────────────

def collect_serpapi_news(symbols: list, api_key: str, verbose: bool = True) -> int:
    """
    Search SerpAPI Google News for each symbol.
    Returns rich results from Google News aggregator (Reuters, FT, Bloomberg, etc.)
    Source name comes from Google's metadata so SOURCE_TIER scoring applies.
    Pricing: ~$0.001–0.002/call depending on plan.
    """
    conn = get_conn()
    saved = 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)

    for sym in symbols:
        try:
            r = requests.get(
                "https://serpapi.com/search",
                params={
                    "api_key": api_key,
                    "engine":  "google_news",
                    "q":       f"{sym} stock",
                    "gl":      "us",
                    "hl":      "en",
                    "num":     10,
                },
                timeout=15,
            )
            if r.status_code != 200:
                if verbose:
                    print(f"[serpapi] {sym}: HTTP {r.status_code}")
                continue

            for item in (r.json().get("news_results") or []):
                title   = (item.get("title") or "").strip()
                url     = item.get("link", "")
                snippet = item.get("snippet", "")
                pub_raw = item.get("date", "")
                src     = item.get("source") or {}
                source_name = src.get("name") or _extract_domain(url)

                if not title or not url:
                    continue

                pub = _parse_relative_date(pub_raw)
                try:
                    if datetime.fromisoformat(pub) < cutoff:
                        continue
                except Exception:
                    pass

                full_text = f"{title} {snippet}"
                tickers   = extract_tickers(full_text, title=title)
                if sym not in tickers:
                    tickers.append(sym)

                event = classify_event(full_text)
                sent  = quick_sentiment(full_text)
                nov   = novelty_score(title, conn)
                imp   = importance_score(event, tickers, abs(sent))

                if _store_article(conn, source_name, title, url, pub,
                                  snippet, tickers, event, sent, nov, imp):
                    saved += 1

        except Exception as e:
            if verbose:
                print(f"[serpapi] {sym}: {e}")

    conn.commit()
    conn.close()
    return saved


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
                tickers   = extract_tickers(full_text, title=title)
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

    # ── Optional: Tavily / SerpAPI per-symbol targeted search ─────────────
    tavily_key  = os.environ.get("TAVILY_API_KEY",  "")
    serpapi_key = os.environ.get("SERPAPI_API_KEY", "")

    if tavily_key or serpapi_key:
        # Top 15 enabled symbols by priority — stays within Tavily free tier
        _c = get_conn()
        _syms = [r["symbol"] for r in _c.execute(
            "SELECT symbol FROM watched_symbols WHERE enabled=1 ORDER BY priority, symbol"
        ).fetchall()][:15]
        _c.close()

        if tavily_key and _syms:
            n = collect_tavily_news(_syms, tavily_key, verbose=verbose)
            saved += n
            if verbose:
                print(f"[news] Tavily:  +{n} articles ({len(_syms)} symbols queried)")

        if serpapi_key and _syms:
            n = collect_serpapi_news(_syms, serpapi_key, verbose=verbose)
            saved += n
            if verbose:
                print(f"[news] SerpAPI: +{n} articles ({len(_syms)} symbols queried)")

    if verbose:
        print(f"[news] collected {saved} articles, skipped {skipped} duplicates")
    return saved
