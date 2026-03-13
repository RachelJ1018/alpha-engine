"""
app.py — Streamlit dashboard for Alpha Engine
Run: streamlit run app.py
"""
import streamlit as st
import sys, os, io, json
from datetime import date
from contextlib import redirect_stdout
from modules.risk_engine import (
    PortfolioConfig, RiskConfig, plan_candidates, candidate_from_row
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Alpha Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Alpha Engine — Daily Investment Research")
st.caption("Local AI-powered trade decision system · Not financial advice")

def _safe(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default
# ── Sidebar: User Configuration ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Your Configuration")

    st.subheader("💰 Portfolio")
    portfolio_value = st.number_input(
        "Total Portfolio Value ($)",
        min_value=500, max_value=10_000_000, value=10_000, step=1_000,
        help="Your total investment portfolio size",
    )
    available_cash = st.number_input(
        "Available Cash to Deploy ($)",
        min_value=0, max_value=10_000_000, value=int(portfolio_value * 0.5), step=500,
        help="Cash ready to invest right now",
    )

    st.subheader("🛡️ Risk Controls")
    daily_risk_budget_pct = st.slider(
        "Daily Risk Budget (% of portfolio)", 0.5, 5.0, 2.0, step=0.25,
        help="Total portfolio % you're willing to lose in a single day across all trades",
    )
    per_trade_risk_pct = st.slider(
        "Per Trade Risk (% of portfolio)", 0.25, 2.0, 0.75, step=0.25,
        help="Max % of portfolio to risk on a single trade (ATR-based stop sizing)",
    )
    max_gross_exposure_pct = st.slider(
        "Max Gross Exposure (% of portfolio)", 20, 100, 70, step=5,
        help="Max total market value of all open positions as % of portfolio",
    )
    stop_loss_pct = st.slider(
        "Fallback Stop Loss (%)", 1, 10, 4,
        help="Used when ATR data is unavailable",
    )
    take_profit_pct = st.slider(
        "Take Profit Target (%)", 2, 30, 8,
        help="Used for portfolio-level R:R display",
    )
    max_positions = st.slider(
        "Max Simultaneous Positions", 1, 10, 3,
        help="Maximum trades open at the same time",
    )

    st.subheader("🎯 Strategy")
    risk_appetite = st.radio(
        "Risk Appetite",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Conservative: score≥70 | Moderate: score≥60 | Aggressive: score≥50",
    )
    allow_shorts = st.checkbox(
        "Allow Short Positions / Puts",
        value=False,
        help="Include bearish / short-side setups",
    )
    trade_style = st.caption("Trade Style: **Swing (1–5 days)**")

    st.subheader("📋 Custom Watchlist")
    custom_tickers_input = st.text_area(
        "Add tickers (one per line or comma-separated)",
        placeholder="HOOD\nRIVN\nPALM",
        help="Added on top of the default watchlist",
    )

    st.divider()
    st.subheader("🔑 Claude API Key")
    api_key_input = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Enables AI-generated trade theses. Without it uses rule-based fallback.",
    )

# ── Derived values ───────────────────────────────────────────────────────
risk_per_trade      = portfolio_value * per_trade_risk_pct / 100
daily_risk_dollars  = portfolio_value * daily_risk_budget_pct / 100
max_single_name_pct = max_gross_exposure_pct / max_positions   # auto-derived
reward_per_trade    = risk_per_trade * (take_profit_pct / stop_loss_pct)
risk_reward_ratio   = take_profit_pct / stop_loss_pct
min_score = {"Conservative": 70, "Moderate": 60, "Aggressive": 50}[risk_appetite]

# ── Top metrics row ──────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Portfolio",          f"${portfolio_value:,.0f}")
m2.metric("Cash Available",     f"${available_cash:,.0f}")
m3.metric("Daily Risk Budget",  f"${daily_risk_dollars:,.0f}",  f"{daily_risk_budget_pct}% of portfolio")
m4.metric("Per Trade Risk",     f"${risk_per_trade:,.0f}",      f"{per_trade_risk_pct}% of portfolio")
m5.metric("Target R:R",         f"1:{risk_reward_ratio:.1f}",   f"stop {stop_loss_pct}% / tp {take_profit_pct}%")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_today, tab_eval = st.tabs(["📊 Today's Research", "🧪 Evaluation"])

# ── Run Button (inside Today tab) ────────────────────────────────────────
with tab_today:
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input

        progress = st.progress(0, text="Starting pipeline…")
        log_box  = st.empty()
        log_lines = []

        def log(msg):
            msg = msg.strip()
            if msg:
                log_lines.append(msg)
                log_box.code("\n".join(log_lines[-25:]), language=None)

        try:
            from modules.db             import init_db, get_conn
            from modules.news_collector import collect_news
            from modules.price_fetcher  import fetch_prices, get_market_regime
            from modules.analyzer       import run_analysis
            from modules.report_generator import generate_report

            today = date.today().isoformat()

            progress.progress(5,  "Initializing database…")
            buf = io.StringIO()
            with redirect_stdout(buf): init_db()
            log(buf.getvalue())

            if custom_tickers_input.strip():
                conn = get_conn()
                raw = custom_tickers_input.replace(",", "\n").split("\n")
                syms = [s.strip().upper() for s in raw if s.strip()]
                for sym in syms:
                    conn.execute(
                        "INSERT OR IGNORE INTO watched_symbols (symbol, sector, priority) VALUES (?,?,?)",
                        (sym, "Custom", 1),
                    )
                conn.commit(); conn.close()
                log(f"Added custom tickers: {', '.join(syms)}")

            progress.progress(15, "Fetching news from RSS feeds…")
            buf = io.StringIO()
            with redirect_stdout(buf): news_count = collect_news()
            log(buf.getvalue())

            progress.progress(45, "Fetching prices & technicals (yfinance)…")
            buf = io.StringIO()
            with redirect_stdout(buf): price_count = fetch_prices()
            log(buf.getvalue())

            conn = get_conn()
            regime = get_market_regime(conn)
            conn.close()
            log(f"Market regime: {regime['regime'].upper()} | SPY {regime.get('spy_change', 0):+.2f}%")

            progress.progress(65, "Running scoring & analysis…")
            buf = io.StringIO()
            with redirect_stdout(buf): candidates_found = run_analysis(regime)
            log(buf.getvalue())

            progress.progress(88, "Generating report…")
            buf = io.StringIO()
            with redirect_stdout(buf): generate_report(regime)
            log(buf.getvalue())

            progress.progress(100, "✅ Done!")
            log(f"Complete — news:{news_count} | prices:{price_count} | candidates:{candidates_found}")

            st.session_state.update({
                "run_complete": True,
                "regime": regime,
                "today": today,
            })

        except Exception as e:
            import traceback
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())

    # ── Results ──────────────────────────────────────────────────────────
    if st.session_state.get("run_complete"):
        today  = st.session_state.get("today",  date.today().isoformat())
        regime = st.session_state.get("regime", {})

        regime_label = regime.get("regime", "unknown")
        spy_chg      = regime.get("spy_change", 0)
        spy_rsi      = regime.get("spy_rsi", None)

        st.subheader("🌍 Market Regime")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Regime",      regime_label.upper())
        rc2.metric("SPY Change",  f"{spy_chg:+.2f}%")
        rc3.metric("SPY RSI(14)", f"{spy_rsi:.1f}" if spy_rsi else "—")

        if regime_label == "bear":
            st.error("📉 BEAR market — reduce sizes, go selective, consider cash or hedges.")
        elif regime_label == "bull":
            st.success("📈 BULL market — favorable for longs. Manage exits.")
        elif regime_label == "choppy":
            st.warning("〰️ CHOPPY — low volume, directionless. Wait for clarity.")
        else:
            st.info("➡️ NEUTRAL — mixed signals. Prefer high-conviction setups only.")

        st.divider()

        from modules.db import get_conn
        conn = get_conn()
        all_candidates = conn.execute("""
            SELECT tc.*, ps.close_price, ps.change_pct, ps.rsi_14,
                   ps.volume_ratio, ps.ma_20, ps.ma_50, ps.atr_14,
                   ws.sector
            FROM trade_candidates tc
            LEFT JOIN price_snapshots ps
              ON tc.symbol = ps.symbol AND ps.snapshot_date = ?
            LEFT JOIN watched_symbols ws ON tc.symbol = ws.symbol
            WHERE tc.run_date = ? AND tc.final_score >= ?
            ORDER BY tc.final_score DESC
        """, (today, today, min_score)).fetchall()
        conn.close()

        if not allow_shorts:
            all_candidates = [c for c in all_candidates if c["direction"] != "SHORT"]

        index_symbols    = {"SPY", "QQQ"}
        market_ideas     = [c for c in all_candidates if c["symbol"] in index_symbols]
        stock_candidates = [c for c in all_candidates if c["symbol"] not in index_symbols]

        risk_cfg = RiskConfig(
            daily_risk_budget_pct=float(daily_risk_budget_pct),
            per_trade_risk_pct=float(per_trade_risk_pct),
            fallback_stop_pct=float(stop_loss_pct),
            atr_multiplier=1.2,
            max_positions=max_positions,
            max_single_name_exposure_pct=float(max_single_name_pct),
            max_gross_exposure_pct=float(max_gross_exposure_pct),
            max_sector_positions=2,
            bull_risk_multiplier=1.0,
            neutral_risk_multiplier=0.8,
            bear_risk_multiplier=0.7,
            choppy_risk_multiplier=0.5,
        )
        portfolio_cfg = PortfolioConfig(
            portfolio_value=float(portfolio_value),
            available_cash=float(available_cash),
            current_gross_exposure=0.0,
        )
        ideas          = [candidate_from_row(dict(c)) for c in stock_candidates]
        plans          = plan_candidates(ideas, portfolio_cfg, risk_cfg, regime_label)
        plans_by_symbol = {p.symbol: p for p in plans}

        actionable = [c for c in all_candidates if c["action"] == "ACTIONABLE"]
        watchlist  = [c for c in all_candidates if c["action"] == "WATCHLIST"]
        monitor    = [c for c in all_candidates if c["action"] == "MONITOR"]

        st.subheader(f"🎯 Research Ideas  ·  score ≥ {min_score}  ·  {risk_appetite}")
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 ACTIONABLE", len(actionable))
        s2.metric("🟡 WATCHLIST",  len(watchlist))
        s3.metric("⚪ MONITOR",    len(monitor))

        if not all_candidates:
            st.info(
                "No research ideas meet your criteria today.\n\n"
                "Try **Aggressive** risk appetite, or check back tomorrow.\n\n"
                f"{'Short positions are disabled.' if not allow_shorts and regime_label == 'bear' else ''}"
            )

        shown = 0
        for c in stock_candidates:
            if shown >= max_positions * 2:
                break

            sym    = c["symbol"]
            score  = c["final_score"]
            action = c["action"]
            direc  = c["direction"]
            price  = c["close_price"] or 0
            chg    = c["change_pct"] or 0
            rsi    = c["rsi_14"]
            vr     = c["volume_ratio"]
            ma20   = c["ma_20"]

            plan = plans_by_symbol.get(sym)
            if plan and plan.allowed:
                shares       = plan.shares
                actual_cost  = plan.position_value
                stop_price   = plan.stop_price
                target_price = plan.target_price
                dollar_risk  = plan.risk_dollars
                dollar_rwd   = plan.reward_dollars
                rr           = plan.rr_ratio
                sizing_note  = (f"×{plan.regime_multiplier} regime · "
                                f"×{plan.action_multiplier} action · "
                                f"×{plan.score_multiplier:.2f} score")
            else:
                shares = actual_cost = stop_price = target_price = dollar_risk = dollar_rwd = rr = 0
                sizing_note = f"BLOCKED — {plan.reason if plan else 'no plan'}"

            action_icon = {"ACTIONABLE": "🟢", "WATCHLIST": "🟡",
                           "MONITOR": "⚪", "IGNORE": "⛔"}.get(action, "⚪")
            dir_icon = "▲" if direc == "LONG" else "▼"
            header   = (f"{action_icon} **{sym}** — Score {score:.0f}/100 | "
                        f"{action} | {dir_icon} {direc} | ${price:.2f} ({chg:+.1f}%)")

            with st.expander(header, expanded=(action == "ACTIONABLE")):
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Shares",  f"{shares:,}",          f"${actual_cost:,.0f} cost")
                p2.metric("Stop",    f"${stop_price:.2f}",   f"-${dollar_risk:,.0f} risk")
                p3.metric("Target",  f"${target_price:.2f}", f"+${dollar_rwd:,.0f} reward")
                p4.metric("R:R",     f"1:{rr:.1f}",          sizing_note)

                st.caption(
                    f"RSI(14): {rsi or '—'} · "
                    f"Vol ratio: {f'{vr:.1f}x' if vr else '—'} · "
                    f"MA20: ${ma20:.2f}" if ma20 else f"RSI(14): {rsi or '—'} · Vol ratio: {f'{vr:.1f}x' if vr else '—'}"
                )
                if c["thesis"]:
                    st.markdown(f"**What & Why:** {c['thesis']}")
                col_a, col_b = st.columns(2)
                with col_a:
                    if c["entry_note"]:     st.markdown(f"**Entry:** {c['entry_note']}")
                    if c["stop_loss_note"]: st.markdown(f"**Stop:** :red[{c['stop_loss_note']}]")
                with col_b:
                    if c["target_note"]:    st.markdown(f"**Target:** :green[{c['target_note']}]")
                    if c["risk_note"]:      st.markdown(f"**If wrong:** :orange[{c['risk_note']}]")
                st.progress(int(score) / 100, text=(
                    f"EventEdge: {_safe(c['event_edge_score']):.1f}/25 · "
                    f"MarketConf: {_safe(c['market_conf_score']):.1f}/20 · "
                    f"RegimeFit: {_safe(c['regime_fit_score']):.1f}/15 · "
                    f"RelOpp: {_safe(c['relative_opp_score']):.1f}/15 · "
                    f"Freshness: {_safe(c['freshness_score']):.1f}/10 · "
                    f"RiskPenalty: -{_safe(c['risk_penalty_score']):.1f}"
                ))
            shown += 1

        # ── News ─────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📰 Key News (Last 24h)")
        from modules.db import get_conn as _gc
        conn2 = _gc()
        top_news = conn2.execute("""
            SELECT * FROM news_articles
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
            LIMIT 12
        """).fetchall()
        conn2.close()

        if top_news:
            for n in top_news:
                sent = n["sentiment_score"] or 0
                icon = "🟢" if sent > 0.1 else "🔴" if sent < -0.1 else "⚪"
                syms = json.loads(n["symbols"] or "[]")
                tickers_str = "  " + "  ".join(f"`{s}`" for s in syms[:3]) if syms else ""
                st.markdown(
                    f"{icon} **{n['title']}**{tickers_str}  \n"
                    f"<small>_{n['source']} · {n['event_type']} · sentiment {sent:+.2f}_</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent news found.")

        if market_ideas:
            st.divider()
            st.subheader("📈 Market Instruments")
            for c in market_ideas:
                st.markdown(
                    f"- **{c['symbol']}** · {c['direction']} · {c['action']} · score {float(c['final_score']):.0f}"
                )

        st.divider()
        st.subheader("🛡️ Risk Configuration")
        st.info(
            f"- Per trade risk: **${risk_per_trade:,.0f}** ({per_trade_risk_pct}% of ${portfolio_value:,.0f})\n"
            f"- Daily budget: **${daily_risk_dollars:,.0f}** ({daily_risk_budget_pct}%) · "
            f"fits {int(daily_risk_dollars // risk_per_trade) if risk_per_trade else 0} full-size trades\n"
            f"- Fallback stop: **{stop_loss_pct}%** · Take profit ref: **+{take_profit_pct}%** · R:R **1:{risk_reward_ratio:.1f}**\n"
            f"- Max positions: **{max_positions}** · max gross exposure: **{max_gross_exposure_pct}%** "
            f"(${portfolio_value * max_gross_exposure_pct / 100:,.0f})\n"
            f"- Regime scaling — Bull: 1.0× · Neutral: 0.8× · Bear: 0.7× · Choppy: 0.5×"
        )

        report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", f"report_{today}.html")
        if os.path.exists(report_path):
            st.caption(f"📄 Full HTML report: `{report_path}`")

# ── Evaluation Tab (always visible) ──────────────────────────────────────
with tab_eval:
    import pandas as pd
    from modules.db import get_conn as _eval_gc
    from modules.evaluator import (
        signal_stability_report, score_return_buckets,
        paper_trade_summary, event_type_breakdown,
        weight_calibration_suggestions,
    )
    from modules.signal_tracker import get_outcome_stats

    eval_days = st.slider("Lookback window (days)", 7, 90, 30, key="eval_days")
    eval_conn = _eval_gc()

    # ── 1. Signal Stability ───────────────────────────────────────────────
    st.subheader("📊 1 — Signal Stability")
    stab = signal_stability_report(eval_conn, days=eval_days)

    if stab.get("total_signals", 0) == 0:
        st.info("No signals found in this window. Run the pipeline first.")
    else:
        ev1, ev2, ev3, ev4 = st.columns(4)
        ev1.metric("Days covered",   stab["days_covered"])
        ev2.metric("Total signals",  stab["total_signals"])
        ev3.metric("Avg / day",      stab["avg_per_day"])
        ev4.metric("Avg score",      stab["avg_score"] or "—")

        ac = stab["action_counts"]
        ac_cols = st.columns(4)
        for i, (label, icon) in enumerate([
            ("ACTIONABLE","🟢"), ("WATCHLIST","🟡"), ("MONITOR","⚪"), ("IGNORE","⛔")
        ]):
            pct = round(ac.get(label, 0) / stab["total_signals"] * 100, 1)
            ac_cols[i].metric(f"{icon} {label}", ac.get(label, 0), f"{pct}%")

        with st.expander("Score breakdown by action"):
            sba = stab["score_by_action"]
            sba_rows = [{"Action": a, "Count": v["count"], "Avg": v["avg"],
                         "Min": v["min"], "Max": v["max"]}
                        for a, v in sba.items()]
            st.dataframe(pd.DataFrame(sba_rows), use_container_width=True, hide_index=True)

        with st.expander("Score component averages"):
            comp = stab["score_components_avg"]
            comp_rows = [
                {"Component": k.replace("_score","").replace("_"," ").title(),
                 "Avg": v,
                 "Max possible": {"event edge":25,"market conf":20,"regime fit":15,
                                  "relative opp":15,"freshness":10,"risk penalty":15}.get(
                     k.replace("_score","").replace("_"," "), "—")}
                for k, v in comp.items() if v is not None
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        with st.expander("Strategy bucket distribution"):
            sd = stab["strategy_distribution"]
            st.dataframe(
                pd.DataFrame([{"Bucket": k, "Count": v} for k, v in sorted(sd.items(), key=lambda x: -x[1])]),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── 2. Score → Return Correlation ────────────────────────────────────
    st.subheader("📈 2 — Score vs Return (does higher score = better P&L?)")
    buckets = score_return_buckets(eval_conn)

    if not buckets:
        st.info("No resolved outcomes yet — needs 5+ trading days of price data after signals.")
    else:
        bkt_rows = [{
            "Score range":    b["bucket"],
            "Signals":        b["count"],
            "Avg t+1 P&L":   f"{b['avg_t1_pnl']:+.2f}%" if b["avg_t1_pnl"] is not None else "—",
            "Avg t+3 P&L":   f"{b['avg_t3_pnl']:+.2f}%" if b["avg_t3_pnl"] is not None else "—",
            "Avg t+5 P&L":   f"{b['avg_t5_pnl']:+.2f}%" if b["avg_t5_pnl"] is not None else "—",
            "Paper P&L":     f"{b['avg_paper_pnl']:+.2f}%" if b["avg_paper_pnl"] is not None else "—",
            "Paper Win%":    f"{b['paper_win_rate']:.0f}%" if b["paper_win_rate"] is not None else "—",
        } for b in buckets]
        st.dataframe(pd.DataFrame(bkt_rows), use_container_width=True, hide_index=True)
        st.caption("Paper Win% = HIT_TARGET + T5_EXIT with positive P&L, over resolved trades.")

    st.divider()

    # ── 3. Paper Trade Summary ────────────────────────────────────────────
    st.subheader("🎯 3 — Paper Trade & Risk Engine Health")
    pt = paper_trade_summary(eval_conn)
    stats_raw = get_outcome_stats(eval_conn)

    if pt.get("total", 0) == 0:
        st.info("No signal outcomes recorded yet.")
    else:
        pt1, pt2, pt3, pt4 = st.columns(4)
        pt1.metric("Total tracked",   pt["total"])
        pt2.metric("Resolved",        pt["resolved"])
        pt3.metric("Paper win rate",  f"{pt['paper_win_rate']}%" if pt["paper_win_rate"] is not None else "—")
        pt4.metric("Avg paper P&L",   f"{pt['avg_paper_pnl']:+.2f}%" if pt["avg_paper_pnl"] is not None else "—")

        if pt["resolved"] > 0:
            ec = pt["exit_counts"]
            ep = pt["exit_pct"]
            ex1, ex2, ex3 = st.columns(3)
            ex1.metric("🛑 HIT_STOP",    ec["HIT_STOP"],    f"{ep['HIT_STOP']}%")
            ex2.metric("🎯 HIT_TARGET",  ec["HIT_TARGET"],  f"{ep['HIT_TARGET']}%")
            ex3.metric("⏱ T5_EXIT",     ec["T5_EXIT"],     f"{ep['T5_EXIT']}%")

        with st.expander("Breakdown by signal level"):
            ab = pt["action_breakdown"]
            ab_rows = [{"Signal": a,
                        "Count": v["count"], "Resolved": v["resolved"],
                        "Avg Score": v["avg_score"] or "—",
                        "Paper Win%": f"{v['paper_win_rate']:.0f}%" if v["paper_win_rate"] is not None else "—",
                        "Avg Paper P&L": f"{v['avg_paper_pnl']:+.2f}%" if v["avg_paper_pnl"] is not None else "—"}
                       for a, v in ab.items()]
            st.dataframe(pd.DataFrame(ab_rows), use_container_width=True, hide_index=True)

        if pt["regime_breakdown"]:
            with st.expander("Breakdown by regime"):
                rb = pt["regime_breakdown"]
                rb_rows = [{"Regime": r,
                            "Count": v["count"],
                            "Paper Win%": f"{v['paper_win_rate']:.0f}%",
                            "Avg Paper P&L": f"{v['avg_paper_pnl']:+.2f}%" if v["avg_paper_pnl"] is not None else "—"}
                           for r, v in rb.items()]
                st.dataframe(pd.DataFrame(rb_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 4. Event Type Breakdown ───────────────────────────────────────────
    st.subheader("🗂 4 — By Event Type (which catalyst actually works?)")
    etb = event_type_breakdown(eval_conn)

    if not etb:
        st.info("No resolved outcomes by event type yet.")
    else:
        et_rows = [{
            "Event type":   r["event_type"],
            "Signals":      r["count"],
            "Resolved":     r["resolved"],
            "Avg t+5 P&L":  f"{r['avg_t5_pnl']:+.2f}%" if r["avg_t5_pnl"] is not None else "—",
            "Paper Win%":   f"{r['paper_win_rate']:.0f}%" if r["paper_win_rate"] is not None else "—",
            "Hit Stop%":    f"{r['hit_stop_pct']:.0f}%" if r["hit_stop_pct"] is not None else "—",
            "Hit Target%":  f"{r['hit_target_pct']:.0f}%" if r["hit_target_pct"] is not None else "—",
        } for r in etb]
        st.dataframe(pd.DataFrame(et_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 5. Weight Calibration Suggestions ────────────────────────────────
    st.subheader("⚖️ 5 — Weight Calibration Suggestions")
    st.caption(
        "Based on resolved paper trades. Compares actual win rate vs current "
        "`EVENT_IMPORTANCE` weights in `analyzer.py`. "
        f"Minimum {3} resolved signals required per event type."
    )
    wcs = weight_calibration_suggestions(eval_conn)

    if not wcs:
        st.info("Not enough resolved outcomes to make suggestions yet.")
    else:
        verdict_icon = {"RAISE": "⬆️", "LOWER": "⬇️", "OK": "✅", "INSUFFICIENT_DATA": "⏳"}
        for row in wcs:
            icon = verdict_icon.get(row["verdict"], "")
            with st.expander(f"{icon} **{row['event_type']}** — {row['verdict']} (weight: {row['current_weight']:.2f}, n={row['count']})"):
                st.markdown(row["suggestion"])
                if row["verdict"] in ("RAISE", "LOWER"):
                    st.code(
                        f"# In analyzer.py, update EVENT_IMPORTANCE:\n"
                        f"EVENT_IMPORTANCE['{row['event_type']}'] = "
                        + (f"{min(row['current_weight'] + 0.10, 1.0):.2f}"
                           if row["verdict"] == "RAISE"
                           else f"{max(row['current_weight'] - 0.10, 0.05):.2f}"),
                        language="python",
                    )

    # ── Signal Outcomes Table ─────────────────────────────────────────────
    st.divider()
    st.subheader("📋 All Signal Outcomes")
    outcomes = eval_conn.execute("""
        SELECT symbol, signal_date, signal, final_score, regime, direction,
               event_type, strategy_bucket, entry_price,
               t1_pnl_pct, t3_pnl_pct, t5_pnl_pct,
               paper_exit, paper_pnl_pct, outcome
        FROM signal_outcomes
        ORDER BY signal_date DESC
        LIMIT 200
    """).fetchall()

    if outcomes:
        def _fmt_pnl(v):
            if v is None: return "—"
            return f"{'🟢' if v > 0.5 else '🔴' if v < -0.5 else '⚪'} {v:+.1f}%"

        out_rows = [{
            "Date":       o["signal_date"],
            "Symbol":     o["symbol"],
            "Signal":     o["signal"],
            "Score":      f"{o['final_score']:.0f}",
            "Regime":     o["regime"],
            "Dir":        o["direction"],
            "Event":      o["event_type"] or "—",
            "Entry":      f"${o['entry_price']:.2f}" if o["entry_price"] else "—",
            "t+1":        _fmt_pnl(o["t1_pnl_pct"]),
            "t+3":        _fmt_pnl(o["t3_pnl_pct"]),
            "t+5":        _fmt_pnl(o["t5_pnl_pct"]),
            "Exit":       o["paper_exit"] or "—",
            "Paper P&L":  _fmt_pnl(o["paper_pnl_pct"]),
            "Outcome":    {"WIN":"✅","LOSS":"❌","SCRATCH":"➖","PENDING":"⏳"}.get(o["outcome"], o["outcome"]),
        } for o in outcomes]
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No signals recorded yet.")

    eval_conn.close()

    st.caption("⚠️ This is a personal research tool, not financial advice. All decisions are your own.")
