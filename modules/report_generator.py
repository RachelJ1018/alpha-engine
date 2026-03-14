"""
report_generator.py — v2
Generates daily_report.md + HTML
Research-first format: fast to scan, explicit layered scoring, less "commanding".
"""

import json
import os
from datetime import date, datetime
from modules.db import get_conn

ACTION_EMOJI = {
    "ACTIONABLE": "🟢",
    "WATCHLIST":  "🟡",
    "MONITOR":    "⚪",
    "IGNORE":     "⛔",
}

ACTION_COLOR = {
    "ACTIONABLE": "#22c55e",
    "WATCHLIST":  "#f59e0b",
    "MONITOR":    "#94a3b8",
    "IGNORE":     "#ef4444",
}

SECTOR_MAP = {
    "AAPL":"tech",  "AMD":"tech",   "AMZN":"tech",  "ARM":"tech",   "AVGO":"tech",
    "GOOGL":"tech", "META":"tech",  "MSFT":"tech",  "NVDA":"tech",  "PLTR":"tech",
    "TSM":"tech",   "SMH":"tech",
    "BAC":"finance","COIN":"finance","GS":"finance", "JPM":"finance","SOFI":"finance",
    "XLF":"finance",
    "GLD":"commodity",
    "IWM":"etf",    "QQQ":"etf",    "SPY":"etf",    "TLT":"bond",
    "COST":"consumer","WMT":"consumer",
    "LLY":"healthcare","UNH":"healthcare",
    "XOM":"energy",
    "LMT":"industrial",
    "TSLA":"auto",
    "NFLX":"media",
}

REGIME_DESC = {
    "bull":    "📈 BULL — Broad market rising. Long setups have better follow-through.",
    "bear":    "📉 BEAR — Market under pressure. Be cautious with longs and size smaller.",
    "neutral": "➡ NEUTRAL — Mixed conditions. Prefer selective, cleaner setups only.",
    "choppy":  "〰 CHOPPY — Directionless and noisy. Avoid chasing; wait for clarity.",
    "unknown": "❓ UNKNOWN — No market data yet.",
}

def normalize_event_type_for_display(news) -> str:
    evt = (news["event_type"] or "general").lower()
    title = (news["title"] or "").lower()
    content = (news["content"] or "").lower()
    text = f"{title} {content}"

    if any(x in text for x in ["gdp", "inflation", "cpi", "ppi", "fed", "powell", "rates"]):
        return "macro"
    if any(x in text for x in ["earnings", "eps", "revenue", "guidance"]):
        return "earnings"
    if "price prediction" in text or "stock prediction" in text:
        return "general"
    return evt

def _safe_num(v, default=0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _score_color(score: float) -> str:
    if score >= 75:
        return "#22c55e"
    if score >= 55:
        return "#f59e0b"
    return "#ef4444"


def _format_tickers(symbols_json: str, limit: int = 3) -> str:
    try:
        syms = json.loads(symbols_json or "[]")
    except Exception:
        syms = []
    return " ".join(f"`{s}`" for s in syms[:limit])


def generate_report(regime: dict, verbose: bool = True) -> str:
    conn = get_conn()
    today = date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    candidates = conn.execute(
        """
        SELECT tc.*, ps.close_price, ps.change_pct, ps.rsi_14,
               ps.volume_ratio, ps.ma_20, ps.ma_50, ps.atr_14
        FROM trade_candidates tc
        LEFT JOIN price_snapshots ps
          ON tc.symbol = ps.symbol AND ps.snapshot_date = ?
        WHERE tc.run_date = ?
        ORDER BY tc.final_score DESC
        LIMIT 20
        """,
        (today, today),
    ).fetchall()

    top_news = conn.execute("""
    SELECT *
    FROM news_articles
    WHERE published_at >= datetime('now', '-24 hours')
      AND LOWER(source) NOT LIKE '%benzinga%'
      AND LOWER(source) NOT LIKE '%seeking alpha%'
      AND LOWER(title) NOT LIKE '%price prediction%'
      AND LOWER(title) NOT LIKE '%stock prediction%'
      AND LOWER(title) NOT LIKE '%should you buy%'
      AND LOWER(title) NOT LIKE '%here''s what%'
      AND LOWER(title) NOT LIKE '%has to say about%'
      AND LOWER(title) NOT LIKE '%think about%'
      AND LOWER(title) NOT LIKE '%thinks about%'
    ORDER BY importance_score DESC, published_at DESC
    LIMIT 10
""").fetchall()

    spy_chg = regime.get("spy_change", 0)
    regime_label = regime.get("regime", "unknown")
    spy_rsi = regime.get("spy_rsi", 50)

    lines = []
    lines.append(f"# 📊 Daily Research Report — {today}")
    lines.append(f"*Generated: {now}*\n")

    lines.append("---")
    lines.append("## 🌍 Market Context\n")
    lines.append(f"**Regime:** {REGIME_DESC.get(regime_label, regime_label)}")
    lines.append(f"**SPY:** {spy_chg:+.2f}%")
    if spy_rsi:
        lines.append(f"**SPY RSI(14):** {spy_rsi}")

    if regime_label == "bear":
        lines.append("\n> ⚠️ **Research posture:** Market is weak. Favor smaller long exposure and avoid chasing strength.")
    elif regime_label == "bull":
        lines.append("\n> ✅ **Research posture:** Market is supportive. Long ideas have a better chance of follow-through.")
    elif regime_label == "choppy":
        lines.append("\n> ⚠️ **Research posture:** Market is noisy. Keep standards high and avoid overtrading.")
    lines.append("")

    action_groups = {}
    for c in candidates:
        action_groups.setdefault(c["action"], []).append(c)

    actionable = action_groups.get("ACTIONABLE", [])
    watchlist = action_groups.get("WATCHLIST", [])
    monitor = action_groups.get("MONITOR", [])
    ignore = action_groups.get("IGNORE", [])

    lines.append("---")
    lines.append("## 📋 Action Summary\n")

    if actionable:
        lines.append(f"**🟢 ACTIONABLE ({len(actionable)}):** " + " · ".join(f"`{c['symbol']}`" for c in actionable))
    else:
        lines.append("**🟢 ACTIONABLE:** None today")

    if watchlist:
        lines.append(f"**🟡 WATCHLIST ({len(watchlist)}):** " + " · ".join(f"`{c['symbol']}`" for c in watchlist))

    if monitor:
        lines.append(f"**⚪ MONITOR ({len(monitor)}):** " + " · ".join(f"`{c['symbol']}`" for c in monitor[:6]))

    if ignore:
        lines.append(f"**⛔ IGNORE ({len(ignore)}):** " + " · ".join(f"`{c['symbol']}`" for c in ignore[:6]))

    lines.append("")

    top_trade = actionable[0] if actionable else (watchlist[0] if watchlist else None)
    if top_trade:
        t = top_trade
        price = _safe_num(t["close_price"])
        atr   = _safe_num(t["atr_14"])
        stop_dist = max(atr * 1.2, price * 0.015)
        if t["direction"] == "LONG":
            stop   = price - stop_dist
            target = price + stop_dist * 2
        else:
            stop   = price + stop_dist
            target = price - stop_dist * 2
        risk_pct   = stop_dist / price * 100 if price > 0 else 0
        reward_pct = stop_dist * 2 / price * 100 if price > 0 else 0

        lines.append("---")
        lines.append("## 🏆 Top Trade\n")
        lines.append(f"**{t['symbol']} — {t['direction']} | {t['action']} | Score: {_safe_num(t['final_score']):.0f}**\n")
        lines.append(f"> {t['thesis'] or '—'}\n")
        lines.append(
            f"| Entry | Stop | Target | Risk | Reward |"
            f"\n|-------|------|--------|------|--------|"
            f"\n| ~${price:.2f} | ${stop:.2f} (-{risk_pct:.1f}%) | ${target:.2f} (+{reward_pct:.1f}%) | 1R | 2R |"
        )
        lines.append("")

    lines.append("---")
    lines.append("## 🎯 Research Ideas\n")
    index_symbols = {"SPY", "QQQ"}
    market_ideas = [c for c in candidates if c["symbol"] in index_symbols]
    stock_ideas = [
        c for c in candidates
        if c["symbol"] not in index_symbols
        and c["action"] in ("ACTIONABLE", "WATCHLIST", "MONITOR")
        and (c["strategy_bucket"] or "") != "macro_watch"
    ]
    macro_watch_ideas = [c for c in candidates if (c["strategy_bucket"] or "") == "macro_watch"]

    sector_counts: dict = {}
    deduped = []
    for c in stock_ideas:
        sec = SECTOR_MAP.get(c["symbol"], "other")
        if sector_counts.get(sec, 0) < 2:
            deduped.append(c)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
    idea_list = deduped
    if not idea_list:
      idea_list = [c for c in candidates if c["action"] in ("ACTIONABLE", "WATCHLIST", "MONITOR") and (c["strategy_bucket"] or "") != "macro_watch"][:5]
    if not idea_list:
      idea_list = [c for c in candidates if (c["strategy_bucket"] or "") != "macro_watch"][:5]

    for i, c in enumerate(idea_list[:8], 1):
        sym = c["symbol"]
        score = _safe_num(c["final_score"])
        action = c["action"]
        direc = c["direction"]
        price = _safe_num(c["close_price"])
        chg = _safe_num(c["change_pct"])
        rsi = c["rsi_14"]
        vr = c["volume_ratio"]
        ma20 = _safe_num(c["ma_20"])
        atr = _safe_num(c["atr_14"])
        emoji = ACTION_EMOJI.get(action, "⚪")
        dir_arrow = "▲" if direc == "LONG" else "▼"

        lines.append(f"### {emoji} #{i} {sym} — {dir_arrow} {direc} | {action} | Score: {score:.0f}/100\n")

        lines.append(
            f"**Price Snapshot:** ${price:.2f} ({chg:+.1f}%) | "
            f"**RSI:** {rsi or '—'} | "
            f"**Vol ratio:** {vr or '—'}x | "
            f"**ATR:** {atr or '—'} | "
            f"**vs MA20:** {'above ✅' if price > (ma20 or 0) else 'below ⚠️'}\n"
        )

        lines.append(
            f"**Layered Scores:** "
            f"EventEdge `{_safe_num(c['event_edge_score']):.1f}/25` | "
            f"MarketConf `{_safe_num(c['market_conf_score']):.1f}/20` | "
            f"RegimeFit `{_safe_num(c['regime_fit_score']):.1f}/15` | "
            f"RelOpp `{_safe_num(c['relative_opp_score']):.1f}/15` | "
            f"Freshness `{_safe_num(c['freshness_score']):.1f}/10` | "
            f"RiskPenalty `-{_safe_num(c['risk_penalty_score']):.1f}`\n"
        )

        strategy_bucket = c["strategy_bucket"] or "—"
        lines.append(f"**Strategy Bucket:** `{strategy_bucket}`\n")

        if c["thesis"]:
            lines.append(f"**What & Why:** {c['thesis']}\n")

        if c["entry_note"]:
            lines.append(f"**Entry Consideration:** {c['entry_note']}")
        if c["stop_loss_note"]:
            lines.append(f"**Risk / Invalidation Level:** {c['stop_loss_note']}")
        if c["target_note"]:
            lines.append(f"**Initial Upside / Downside Case:** {c['target_note']}")
        if c["risk_note"]:
            lines.append(f"**Key Risk:** ❌ {c['risk_note']}")

        lines.append("")

    if macro_watch_ideas:
        lines.append("---")
        lines.append("## 🌍 Macro Watchlist\n")
        for c in macro_watch_ideas:
            emoji = ACTION_EMOJI.get(c["action"], "⚪")
            lines.append(f"- {emoji} `{c['symbol']}` {c['direction']} | {c['action']} | score {c['final_score']:.0f} | {(c['thesis'] or '—')}")
        lines.append("")

    if market_ideas:
      lines.append("---")
      lines.append("## 📈 Market Instruments\n")
      for c in market_ideas:
          lines.append(f"- `{c['symbol']}` {c['direction']} | {c['action']} | score {c['final_score']:.0f}")
      lines.append("")

    lines.append("---")
    lines.append("## 📰 Key News (Last 24h)\n")

    for news in top_news[:8]:
        sent = _safe_num(news["sentiment_score"])
        emoji = "🟢" if sent > 0.1 else "🔴" if sent < -0.1 else "⚪"
        tickers_str = _format_tickers(news["symbols"], limit=3)
        lines.append(f"{emoji} **[{news['source']}]** {news['title']}")
        if tickers_str:
            lines.append(f"   *Tickers: {tickers_str}*")
        display_evt = normalize_event_type_for_display(news)
        lines.append(
              f"   *{display_evt} · sentiment: {sent:+.2f} · "
              f"importance: {_safe_num(news['importance_score']):.2f} · "
              f"novelty: {_safe_num(news['novelty_score']):.2f}*\n"
          )

    lines.append("---")
    lines.append("## 🛡 Research Risk Rules\n")
    lines.append("- Use predefined risk limits **before** entering any trade.")
    lines.append("- Favor **smaller size** in bear or choppy regimes.")
    lines.append("- Avoid chasing names already **extended** on the day.")
    lines.append("- Keep total active ideas limited; focus on the **best few**.")
    lines.append("- If broad market conditions deteriorate sharply, reassess all long theses.")
    lines.append("")

    report_md = "\n".join(lines)

    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)

    md_path = os.path.join(report_dir, f"report_{today}.md")
    html_path = os.path.join(report_dir, f"report_{today}.html")
    latest_md = os.path.join(report_dir, "latest.md")

    with open(md_path, "w") as f:
        f.write(report_md)
    with open(latest_md, "w") as f:
        f.write(report_md)

    html = build_html(today, regime, idea_list, top_news, macro_watch_ideas)
    with open(html_path, "w") as f:
        f.write(html)

    conn.close()
    if verbose:
        print(f"[report] saved → {md_path}")
        print(f"[report] HTML  → {html_path}")
    return report_md


def _macro_watch_html(macro_watch_ideas):
    items = " · ".join(
        f'<code>{c["symbol"]}</code> {c["action"]} | {c["final_score"]:.0f}'
        for c in macro_watch_ideas
    )
    return f'''
<h2>🌍 Macro Watchlist</h2>
<div class="card" style="border-left:3px solid #64748b">
  <div style="font-size:12px;color:#94a3b8">{items}</div>
</div>
'''


def build_html(today, regime, candidates, news_items, macro_watch_ideas=None):
    regime_label = regime.get("regime", "neutral")
    spy_chg = regime.get("spy_change", 0)
    regime_color = {
        "bull": "#22c55e",
        "bear": "#ef4444",
        "neutral": "#f59e0b",
        "choppy": "#94a3b8",
    }.get(regime_label, "#94a3b8")

    top_trade = next((c for c in candidates if c["action"] == "ACTIONABLE"), None)
    if not top_trade:
        top_trade = next((c for c in candidates if c["action"] == "WATCHLIST"), None)

    top_trade_html = ""
    if top_trade:
        t = top_trade
        tt_price = _safe_num(t["close_price"])
        tt_atr   = _safe_num(t["atr_14"])
        tt_stop_dist = max(tt_atr * 1.2, tt_price * 0.015)
        if t["direction"] == "LONG":
            tt_stop   = tt_price - tt_stop_dist
            tt_target = tt_price + tt_stop_dist * 2
        else:
            tt_stop   = tt_price + tt_stop_dist
            tt_target = tt_price - tt_stop_dist * 2
        tt_risk_pct   = tt_stop_dist / tt_price * 100 if tt_price > 0 else 0
        tt_reward_pct = tt_stop_dist * 2 / tt_price * 100 if tt_price > 0 else 0
        tt_action_color = ACTION_COLOR.get(t["action"], "#94a3b8")
        tt_bg    = "#1e3a5f" if t["direction"] == "LONG" else "#3a1e1e"
        tt_dir_color = "#22c55e" if t["direction"] == "LONG" else "#ef4444"
        tt_score = _safe_num(t["final_score"])
        tt_score_color = _score_color(tt_score)
        top_trade_html = f"""
<div class="card" style="background:{tt_bg};border-left:3px solid {tt_action_color}">
  <div class="card-header">
    <div>
      <span class="ticker">{t["symbol"]}</span>
      <span class="action-badge" style="background:{tt_action_color}22;color:{tt_action_color}">{t["action"]}</span>
      <span class="dir-badge" style="color:{tt_dir_color}">{"&#9650; LONG" if t["direction"]=="LONG" else "&#9660; SHORT"}</span>
    </div>
    <div class="score-box">
      <div style="font-size:22px;font-weight:700;color:{tt_score_color}">{tt_score:.0f}</div>
      <div style="font-size:10px;color:#64748b">/ 100</div>
    </div>
  </div>
  <div class="thesis" style="margin-bottom:14px">{t["thesis"] or "—"}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:12px">
    <div style="text-align:center;background:#0b0f1a;border-radius:4px;padding:8px">
      <div style="color:#475569;font-size:9px;letter-spacing:.1em">ENTRY</div>
      <div style="color:#f1f5f9;font-weight:600">~${tt_price:.2f}</div>
    </div>
    <div style="text-align:center;background:#0b0f1a;border-radius:4px;padding:8px">
      <div style="color:#475569;font-size:9px;letter-spacing:.1em">STOP</div>
      <div style="color:#f87171;font-weight:600">${tt_stop:.2f} (-{tt_risk_pct:.1f}%)</div>
    </div>
    <div style="text-align:center;background:#0b0f1a;border-radius:4px;padding:8px">
      <div style="color:#475569;font-size:9px;letter-spacing:.1em">TARGET</div>
      <div style="color:#22c55e;font-weight:600">${tt_target:.2f} (+{tt_reward_pct:.1f}%)</div>
    </div>
  </div>
</div>
"""

    cards_html = ""
    for c in candidates[:8]:
        sym = c["symbol"]
        score = _safe_num(c["final_score"])
        action = c["action"]
        direc = c["direction"]
        price = _safe_num(c["close_price"])
        chg = _safe_num(c["change_pct"])
        rsi = c["rsi_14"] or "—"
        vr = c["volume_ratio"] or "—"

        action_color = ACTION_COLOR.get(action, "#94a3b8")
        dir_color = "#22c55e" if direc == "LONG" else "#ef4444"
        score_color = _score_color(score)
        score_bar = int(max(0, min(score, 100)))

        cards_html += f"""
<div class="card" style="border-left:3px solid {action_color}">
  <div class="card-header">
    <div>
      <span class="ticker">{sym}</span>
      <span class="action-badge" style="background:{action_color}22;color:{action_color}">{action.replace('_',' ')}</span>
      <span class="dir-badge" style="color:{dir_color}">{"▲ LONG" if direc=="LONG" else "▼ SHORT"}</span>
    </div>
    <div class="score-box">
      <div style="font-size:22px;font-weight:700;color:{score_color}">{score:.0f}</div>
      <div style="font-size:10px;color:#64748b">/ 100</div>
    </div>
  </div>

  <div class="price-row">
    ${price:.2f}
    <span style="color:{'#22c55e' if chg >= 0 else '#ef4444'}">{chg:+.1f}%</span>
    &nbsp;·&nbsp; RSI: {rsi}
    &nbsp;·&nbsp; Vol: {vr}x
  </div>

  <div class="score-bar-wrap">
    <div class="score-bar" style="width:{score_bar}%;background:{score_color}"></div>
  </div>

  <div class="section-label">LAYERED SCORES</div>
  <div class="score-detail">
    EventEdge: {_safe_num(c["event_edge_score"]):.1f}/25 ·
    MarketConf: {_safe_num(c["market_conf_score"]):.1f}/20 ·
    RegimeFit: {_safe_num(c["regime_fit_score"]):.1f}/15 ·
    RelOpp: {_safe_num(c["relative_opp_score"]):.1f}/15 ·
    Freshness: {_safe_num(c["freshness_score"]):.1f}/10 ·
    RiskPenalty: -{_safe_num(c["risk_penalty_score"]):.1f}
  </div>

  <div class="section-label">STRATEGY</div>
  <div class="meta-chip">{c["strategy_bucket"] or "—"}</div>

  <div class="section-label">WHAT & WHY</div>
  <div class="thesis">{c["thesis"] or "—"}</div>

  <div class="levels">
    <div class="level-row"><span class="level-label">Entry</span><span class="level-val">{c["entry_note"] or "—"}</span></div>
    <div class="level-row"><span class="level-label">Invalidation</span><span class="level-val stop">{c["stop_loss_note"] or "—"}</span></div>
    <div class="level-row"><span class="level-label">Initial Case</span><span class="level-val target">{c["target_note"] or "—"}</span></div>
    <div class="level-row"><span class="level-label">Key Risk</span><span class="level-val risk">{c["risk_note"] or "—"}</span></div>
  </div>
</div>
"""

    news_html = ""
    for n in (news_items or [])[:6]:
        sent = _safe_num(n["sentiment_score"])
        sc = "#22c55e" if sent > 0.1 else "#ef4444" if sent < -0.1 else "#94a3b8"
        display_evt = normalize_event_type_for_display(n)
        try:
            syms = json.loads(n["symbols"] or "[]")
        except Exception:
            syms = []

        chips = ""
        if syms:
            chips = " · " + " ".join(f'<code>{s}</code>' for s in syms[:3])

        news_html += f"""
<div class="news-item">
  <div style="display:flex;gap:8px;align-items:flex-start">
    <span style="color:{sc};font-size:14px;flex-shrink:0">{"▲" if sent > 0.1 else "▼" if sent < -0.1 else "◆"}</span>
    <div>
      <div class="news-title">{n["title"]}</div>
      <div class="news-meta">
        {n["source"]} · {display_evt} · sent {sent:+.2f}
        · imp {_safe_num(n["importance_score"]):.2f}
        · nov {_safe_num(n["novelty_score"]):.2f}
        {chips}
      </div>
    </div>
  </div>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Report — {today}</title>
<style>
  * {{box-sizing:border-box;margin:0;padding:0}}
  body {{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    background:#0b0f1a;color:#e2e8f0;padding:24px;max-width:920px;margin:0 auto;line-height:1.5
  }}
  h1 {{font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:4px}}
  h2 {{font-size:14px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;margin:28px 0 12px}}
  .meta {{font-size:12px;color:#475569;margin-bottom:24px}}
  .regime-bar {{
    background:#111827;border:.5px solid #1e293b;border-left:3px solid {regime_color};
    border-radius:6px;padding:12px 16px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center
  }}
  .regime-label {{font-size:13px;font-weight:600;color:{regime_color}}}
  .spy-chg {{font-size:20px;font-weight:700;color:{"#22c55e" if spy_chg>=0 else "#ef4444"}}}

  .card {{
    background:#111827;border:.5px solid #1e293b;border-radius:8px;padding:18px 20px;margin-bottom:12px
  }}
  .card-header {{
    display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px
  }}
  .ticker {{font-size:20px;font-weight:800;color:#f1f5f9;margin-right:10px;letter-spacing:.04em}}
  .action-badge {{font-size:10px;font-weight:700;padding:3px 8px;border-radius:3px;letter-spacing:.06em;margin-right:6px}}
  .dir-badge {{font-size:12px;font-weight:700}}
  .score-box {{text-align:center}}
  .price-row {{font-size:13px;color:#94a3b8;margin-bottom:8px}}
  .score-bar-wrap {{height:3px;background:#1e293b;border-radius:2px;margin-bottom:14px}}
  .score-bar {{height:3px;border-radius:2px;transition:width .5s}}

  .section-label {{
    font-size:9px;font-weight:700;color:#475569;letter-spacing:.1em;margin-bottom:5px;margin-top:10px
  }}
  .score-detail {{
    font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:12px;padding-left:10px;border-left:.5px solid #1e293b
  }}
  .meta-chip {{
    display:inline-block;background:#1e293b;color:#cbd5e1;font-size:11px;padding:3px 8px;border-radius:999px;margin-bottom:12px
  }}
  .thesis {{
    font-size:13px;color:#cbd5e1;line-height:1.6;margin-bottom:14px;padding-left:10px;border-left:.5px solid #1e293b
  }}

  .levels {{display:flex;flex-direction:column;gap:6px}}
  .level-row {{display:flex;gap:12px;font-size:12px;align-items:baseline}}
  .level-label {{color:#475569;min-width:74px;font-size:10px;font-weight:600;letter-spacing:.06em}}
  .level-val {{color:#94a3b8}}
  .level-val.stop {{color:#f87171}}
  .level-val.target {{color:#22c55e}}
  .level-val.risk {{color:#fb923c;font-style:italic}}

  .news-item {{border-bottom:.5px solid #1e293b;padding:10px 0}}
  .news-item:last-child {{border-bottom:none}}
  .news-title {{font-size:13px;color:#cbd5e1;line-height:1.4;margin-bottom:3px}}
  .news-meta {{font-size:11px;color:#475569}}
  .news-meta code {{background:#1e293b;color:#94a3b8;padding:1px 5px;border-radius:2px;font-size:10px}}

  .risk-box {{background:#111827;border:.5px solid #1e293b;border-left:3px solid #ef4444;border-radius:6px;padding:14px 18px}}
  .risk-box li {{font-size:13px;color:#94a3b8;margin-bottom:6px;padding-left:4px}}
</style>
</head>
<body>
<h1>📊 Daily Research Report</h1>
<div class="meta">Generated: {today} — Alpha Engine Local</div>

<div class="regime-bar">
  <div>
    <div style="font-size:10px;color:#475569;margin-bottom:3px">MARKET REGIME</div>
    <div class="regime-label">{regime_label.upper()}</div>
    <div style="font-size:12px;color:#64748b;margin-top:2px">
      {REGIME_DESC.get(regime_label,"").split("—")[1].strip() if "—" in REGIME_DESC.get(regime_label,"") else ""}
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:10px;color:#475569;margin-bottom:3px">SPY TODAY</div>
    <div class="spy-chg">{spy_chg:+.2f}%</div>
  </div>
</div>

<h2>🏆 Top Trade</h2>
{top_trade_html if top_trade_html else '<div class="card" style="color:#64748b;font-size:13px">No actionable or watchlist trade today.</div>'}

<h2>🎯 Research Ideas</h2>
{cards_html if cards_html else '<div class="card">No research ideas generated today.</div>'}

{_macro_watch_html(macro_watch_ideas) if macro_watch_ideas else ''}

<h2>📰 Key News</h2>
<div style="background:#111827;border:.5px solid #1e293b;border-radius:8px;padding:14px 18px">
{news_html if news_html else '<div class="news-meta">No recent news items available.</div>'}
</div>

<h2>🛡 Research Risk Rules</h2>
<div class="risk-box">
  <ul style="list-style:none;padding:0">
    <li>📌 Define risk limits before entering any position.</li>
    <li>⚠ Prefer smaller size in bear or choppy regimes.</li>
    <li>🚫 Avoid chasing names already extended on the day.</li>
    <li>🎯 Focus on the best few ideas rather than too many weak ones.</li>
    <li>🌍 Reassess long theses if broad market conditions deteriorate sharply.</li>
  </ul>
</div>

<div style="margin-top:24px;font-size:11px;color:#334155;border-top:.5px solid #1e293b;padding-top:16px">
  ⚠ This is a personal research tool, not financial advice.
</div>
</body>
</html>"""