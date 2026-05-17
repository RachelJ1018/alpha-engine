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
from modules.multi_agent_thesis import _llm_call
from modules.analyzer import compute_catalyst_why
from modules.decision_card import (
    get_historical_evidence, confirmation_lights,
    generate_invalidation, verdict, evidence_interpretation,
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

def _llm_chat(context: str, messages: list, provider: str = "auto") -> str:
    """Follow-up chat using existing multi_agent_thesis._llm_call routing."""
    prompt = f"You are an equity research assistant.\n\nContext:\n{context}\n\n" + \
             "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages[:-1]) + \
             f"\n\nUser: {messages[-1]['content']}\n\nAssistant:"
    text, _ = _llm_call(prompt, provider)
    return text or "No response from LLM."
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
    st.subheader("🔑 API Keys")
    google_api_key_input = st.text_input(
        "Google API Key (Gemini · recommended)",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Free tier: 1500 req/day. Used for thesis generation and chat.",
    )
    api_key_input = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Fallback if Google key not set.",
    )
    thesis_provider = st.selectbox(
        "Thesis Provider",
        ["google", "auto", "groq", "anthropic", "none"],
        index=0,
        help="google = Gemini Flash (recommended). auto = tries Anthropic→Groq→Gemini.",
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
        if google_api_key_input:
            os.environ["GOOGLE_API_KEY"] = google_api_key_input
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
        os.environ["THESIS_PROVIDER"] = thesis_provider

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

        evidence_conn = get_conn()   # shared across all cards; closed after loop

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
            try:
                cq = c["catalyst_quality"] or "NONE"
            except (IndexError, KeyError):
                cq = "NONE"
            cq_badge = {"STRONG": "🔥 STRONG catalyst", "MEDIUM": "✅ MEDIUM catalyst",
                        "WEAK": "⚠️ WEAK catalyst",    "NONE":   "📊 No catalyst"}.get(cq, cq)
            header   = (f"{action_icon} **{sym}** — Score {score:.0f}/100 | "
                        f"{action} | {cq_badge} | {dir_icon} {direc} | ${price:.2f} ({chg:+.1f}%)")

            with st.expander(header, expanded=(action == "ACTIONABLE")):

                # ── Shared card data ─────────────────────────────────────
                _bucket = c["strategy_bucket"] or ""
                _evt_map = {
                    "post_earnings_drift": "earnings", "pre_earnings_drift": "earnings",
                    "event_long": "earnings", "event_short": "earnings",
                    "macro_watch": "macro", "sympathy_play": "general",
                    "opinion_watch": "general", "general_setup": "general",
                    "mean_reversion_long": "general", "relative_strength_long": "general",
                }
                _evt = _evt_map.get(_bucket, "general")
                _earn_str = float(c["earn_strength"] or 0)
                _mc = _safe(c["market_conf_score"])
                _rp = _safe(c["risk_penalty_score"])

                # Historical evidence (uses shared evidence_conn)
                _evidence = get_historical_evidence(
                    evidence_conn, _bucket, direc,
                    float(vr or 1.0), _mc, regime_label,
                )
                _ev_level = _evidence["evidence_level"]

                # Verdict
                _verdict = verdict(action, cq, _ev_level)
                st.markdown(f"**Verdict:** {_verdict}")
                st.divider()

                # ── Section 1: What happened? ─────────────────────────────
                st.markdown("**1. What happened?**")
                _why = compute_catalyst_why(
                    event_type=_evt,
                    catalyst_quality=cq,
                    direction=direc,
                    change_pct=float(chg or 0),
                    volume_ratio=float(vr or 1.0),
                    earn_strength=_earn_str,
                    strategy_bucket=_bucket,
                )
                st.markdown(_why)

                # ── Section 2: Why it may work? ───────────────────────────
                st.markdown("**2. Why it may work?**")
                if c["thesis"]:
                    st.markdown(c["thesis"])
                else:
                    st.caption("No LLM thesis available for this signal.")

                # ── Section 3: Historical evidence ────────────────────────
                st.markdown("**3. Historical Evidence**")
                st.caption(f"Similar setup: _{_evidence['match_desc']}_")
                _ev_color = {
                    "RELIABLE_POSITIVE":   "green",
                    "DEVELOPING_POSITIVE": "green",
                    "EARLY_POSITIVE":      "orange",
                    "EARLY_WEAK":          "orange",
                    "DEVELOPING_WEAK":     "red",
                    "RELIABLE_WEAK":       "red",
                    "INSUFFICIENT":        "grey",
                }.get(_ev_level, "grey")
                if _evidence["n"] >= 3:
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("n",         _evidence["n"])
                    e2.metric("Win rate",  f"{_evidence['win_rate']}%")
                    e3.metric("Avg T+5",   f"{_evidence['avg_t5']:+.2f}%")
                    e4.metric("Worst T+5", f"{_evidence['worst_t5']:+.2f}%")
                    e5, e6, _ = st.columns(3)
                    e5.metric("Hit stop",   f"{_evidence['hit_stop_rate']}%")
                    e6.metric("Hit target", f"{_evidence['hit_target_rate']}%")
                    st.markdown(f"**Evidence level:** :{_ev_color}[{_ev_level}]")
                    st.caption(_evidence["interpretation"])
                else:
                    st.caption(f":{_ev_color}[{_ev_level}] — {_evidence['interpretation']}")

                # ── Section 4: Confirmation ───────────────────────────────
                st.markdown("**4. Confirmation**")
                _lights = confirmation_lights(cq, float(vr or 0), _mc, regime_label, direc, _rp)
                for _label, (_color, _icon, _desc) in _lights.items():
                    st.markdown(f"{_icon} **{_label}:** {_desc}")

                # ── Section 5: Trade plan ─────────────────────────────────
                st.markdown("**5. Trade plan**")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Entry",  f"${price:.2f}",       "next open / current price")
                t2.metric("Stop",   f"${stop_price:.2f}",  f"-${dollar_risk:,.0f} risk")
                t3.metric("Target", f"${target_price:.2f}",f"+${dollar_rwd:,.0f} reward")
                t4.metric("R:R",    f"1:{rr:.1f}",          "T+5 time exit")
                if c["entry_note"]:     st.markdown(f"Entry note: {c['entry_note']}")
                if c["stop_loss_note"]: st.markdown(f":red[Stop: {c['stop_loss_note']}]")
                if c["target_note"]:    st.markdown(f":green[Target: {c['target_note']}]")

                st.markdown("**Do not trade if:**")
                _inv_rules = generate_invalidation(direc, _bucket, sym, stop_price or None)
                for _rule in _inv_rules:
                    st.markdown(f"- {_rule}")

                # ── Section 6: Position sizing ────────────────────────────
                st.markdown("**6. Position sizing**")
                _pos_mult = float(c["position_size_mult"] or 1.0)
                _eff_risk = float(per_trade_risk_pct) * _pos_mult
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Shares",        f"{shares:,}",          f"${actual_cost:,.0f} notional")
                p2.metric("Max loss",      f"${dollar_risk:,.0f}", f"{_eff_risk:.3f}% of portfolio")
                p3.metric("Size mult",     f"{_pos_mult:.2f}×",    sizing_note)
                p4.metric("Base risk",     f"{per_trade_risk_pct}%", f"→ eff. {_eff_risk:.3f}%")
                if shares == 0:
                    st.warning(f"Position blocked: {sizing_note}")

                # ── Score breakdown (transparency) ────────────────────────
                st.progress(min(score / 85, 1.0), text=(
                    f"EventEdge: {_safe(c['event_edge_score']):.1f}/25 · "
                    f"MarketConf: {_safe(c['market_conf_score']):.1f}/20 · "
                    f"RegimeFit: {_safe(c['regime_fit_score']):.1f}/15 · "
                    f"RelOpp: {_safe(c['relative_opp_score']):.1f}/15 · "
                    f"Freshness: {_safe(c['freshness_score']):.1f}/10 · "
                    f"RiskPenalty: -{_safe(c['risk_penalty_score']):.1f}"
                ))
                st.caption(
                    f"RSI(14): {rsi or '—'} · "
                    f"Vol: {f'{vr:.2f}x' if vr else '—'} · "
                    f"MA20: ${ma20:.2f}" if ma20 else
                    f"RSI(14): {rsi or '—'} · Vol: {f'{vr:.2f}x' if vr else '—'}"
                )

                # ── Inline chat ───────────────────────────────────────────
                _ctx = (
                    f"Symbol: {sym} | Tier: {action} | Score: {score:.1f}/85 | "
                    f"Catalyst: {cq} | Evidence: {_ev_level}\n"
                    f"Direction: {direc} | Bucket: {_bucket} | Regime: {regime_label}\n"
                    f"Price: ${price:.2f} ({chg:+.1f}%) | RSI: {rsi or '—'} | Vol: {vr or '—'}x\n"
                    f"Entry: ${price:.2f} | Stop: ${stop_price:.2f} | Target: ${target_price:.2f}\n"
                    f"Thesis: {c['thesis'] or '(none)'}\n"
                    f"Risk note: {c['risk_note'] or '—'}\n"
                    f"Historical n={_evidence['n']} win={_evidence['win_rate']}% avg_t5={_evidence['avg_t5']:+.2f}%"
                )
                _chat_key = f"chat_{sym}_{today}"
                if _chat_key not in st.session_state:
                    st.session_state[_chat_key] = []
                for _msg in st.session_state[_chat_key]:
                    st.chat_message(_msg["role"]).write(_msg["content"])
                _q = st.chat_input(f"Ask about {sym}…", key=f"input_{sym}_{today}")
                if _q:
                    st.session_state[_chat_key].append({"role": "user", "content": _q})
                    with st.spinner("Thinking…"):
                        _ans = _llm_chat(_ctx, st.session_state[_chat_key], thesis_provider)
                    st.session_state[_chat_key].append({"role": "assistant", "content": _ans})
                    st.rerun()
            shown += 1

        evidence_conn.close()

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
        benchmark_adjusted_return, deduplicated_event_return,
        r_multiple_analysis, win_rate_confidence_intervals,
        component_correlation_report, rescore_comparison_report,
        promoted_signal_quality_report, false_upgrade_diagnosis_report,
        empirical_threshold_backtest,
    )
    from modules.signal_tracker import get_outcome_stats

    eval_days = st.slider("Lookback window (days)", 7, 90, 30, key="eval_days")
    eval_conn = _eval_gc()

    # ── 0. Pipeline Health ────────────────────────────────────────────────
    st.subheader("🔧 0 — Pipeline Health")
    _runs = eval_conn.execute("""
        SELECT run_date, run_at, news_fetched, prices_fetched, candidates_found,
               market_regime, spy_change_pct, steps_json
        FROM daily_runs ORDER BY run_date DESC LIMIT 14
    """).fetchall()

    if not _runs:
        st.info("No pipeline runs recorded yet.")
    else:
        _run_rows = []
        for _r in _runs:
            _steps = {}
            try:
                _steps = {s["step"]: s for s in json.loads(_r["steps_json"] or "[]")}
            except Exception:
                pass

            def _fmt_step(name, _steps=_steps):
                if name not in _steps:
                    return "—"
                s = _steps[name]
                return f"{'✅' if s['ok'] else '❌'} {s['ms']}ms"

            _total_ms = sum(s.get("ms", 0) for s in _steps.values())
            _run_rows.append({
                "Date":     _r["run_date"],
                "Regime":   _r["market_regime"] or "—",
                "SPY":      f"{_r['spy_change_pct']:+.2f}%" if _r["spy_change_pct"] is not None else "—",
                "News":     _fmt_step("news"),
                "Prices":   _fmt_step("prices"),
                "Analysis": _fmt_step("analysis"),
                "Report":   _fmt_step("report"),
                "Total":    f"{_total_ms//1000}s" if _total_ms > 0 else f"{_r['candidates_found'] or 0} cands",
            })
        st.dataframe(pd.DataFrame(_run_rows), use_container_width=True, hide_index=True)

    st.divider()

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

    # ── 6. Benchmark-Adjusted Return ─────────────────────────────────────
    st.divider()
    st.subheader("📉 6 — Direction-Adjusted Alpha (vs SPY / sector ETF, same direction)")
    st.caption(
        "**dir_adj_alpha** = signal return − same-direction benchmark return. "
        "LONG: signal − benchmark_long. SHORT: signal − benchmark_short (benchmark inverted). "
        "This is NOT 'signal vs holding SPY long' — it measures whether the signal beat "
        "a same-direction bet on the benchmark."
    )
    bm = benchmark_adjusted_return(eval_conn)

    if not bm or not bm.get("overall"):
        st.info("Not enough resolved outcomes with price data for benchmark comparison.")
    else:
        ov = bm["overall"]
        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Signals with benchmark", ov["n"])
        bm2.metric("Dir-adj alpha t+1", f"{ov['avg_dir_adj_alpha_t1']:+.2f}%" if ov["avg_dir_adj_alpha_t1"] is not None else "—")
        bm3.metric("Dir-adj alpha t+3", f"{ov['avg_dir_adj_alpha_t3']:+.2f}%" if ov["avg_dir_adj_alpha_t3"] is not None else "—")
        bm4.metric("Dir-adj alpha t+5", f"{ov['avg_dir_adj_alpha_t5']:+.2f}%" if ov["avg_dir_adj_alpha_t5"] is not None else "—")

        col_sec, col_reg = st.columns(2)
        with col_sec:
            with st.expander("Dir-adj alpha by sector"):
                sec_rows = [
                    {"Sector":          sec,
                     "n":               v["n"],
                     "Signal t+5":      f"{v['avg_signal_t5']:+.2f}%"        if v["avg_signal_t5"]        is not None else "—",
                     "Bench t+5":       f"{v['avg_bench_t5']:+.2f}%"         if v["avg_bench_t5"]         is not None else "—",
                     "Dir-adj alpha":   f"{v['avg_dir_adj_alpha_t5']:+.2f}%" if v["avg_dir_adj_alpha_t5"] is not None else "—"}
                    for sec, v in sorted(bm["by_sector"].items(), key=lambda x: -(x[1]["avg_dir_adj_alpha_t5"] or -99))
                ]
                st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)
        with col_reg:
            with st.expander("Dir-adj alpha by regime"):
                reg_rows = [
                    {"Regime":        reg,
                     "n":             v["n"],
                     "Dir-adj alpha": f"{v['avg_dir_adj_alpha_t5']:+.2f}%" if v["avg_dir_adj_alpha_t5"] is not None else "—"}
                    for reg, v in bm["by_regime"].items()
                ]
                st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

        col_best, col_worst = st.columns(2)
        with col_best:
            with st.expander("Top 5 best dir-adj alpha"):
                st.dataframe(pd.DataFrame(bm["best"]), use_container_width=True, hide_index=True)
        with col_worst:
            with st.expander("Top 5 worst dir-adj alpha"):
                st.dataframe(pd.DataFrame(bm["worst"]), use_container_width=True, hide_index=True)

    st.divider()

    # ── 7. De-duplicated Event Return ─────────────────────────────────────
    st.subheader("🧹 7 — De-duplicated Event Return")
    st.caption("Same ticker + event type + ISO week counts as one signal. Shows whether returns are inflated by counting the same catalyst multiple times.")
    dd = deduplicated_event_return(eval_conn)

    if not dd:
        st.info("Not enough resolved outcomes for deduplication analysis.")
    else:
        dd1, dd2, dd3 = st.columns(3)
        dd1.metric("Raw signals",    dd["raw_count"])
        dd2.metric("After dedup",    dd["dedup_count"], f"−{dd['removed_pct']}% removed")
        dd3.metric("Dedup win rate", f"{dd['dedup_stats']['paper_win_rate']:.0f}%" if dd["dedup_stats"]["paper_win_rate"] is not None else "—",
                   delta=f"{(dd['dedup_stats']['paper_win_rate'] or 0) - (dd['raw_stats']['paper_win_rate'] or 0):+.1f}% vs raw")

        rs, ds = dd["raw_stats"], dd["dedup_stats"]
        comp_rows = [
            {"Metric": "Avg t+5 P&L",    "Raw": f"{rs['avg_t5_pnl']:+.2f}%"    if rs["avg_t5_pnl"]    is not None else "—",
                                          "De-duped": f"{ds['avg_t5_pnl']:+.2f}%"    if ds["avg_t5_pnl"]    is not None else "—"},
            {"Metric": "Avg paper P&L",  "Raw": f"{rs['avg_paper_pnl']:+.2f}%" if rs["avg_paper_pnl"] is not None else "—",
                                          "De-duped": f"{ds['avg_paper_pnl']:+.2f}%" if ds["avg_paper_pnl"] is not None else "—"},
            {"Metric": "Paper win rate", "Raw": f"{rs['paper_win_rate']:.0f}%"  if rs["paper_win_rate"] is not None else "—",
                                          "De-duped": f"{ds['paper_win_rate']:.0f}%"  if ds["paper_win_rate"] is not None else "—"},
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        with st.expander("By event type — raw vs de-duped"):
            et_dd_rows = [{
                "Event":          r["event_type"],
                "Raw n":          r["raw_n"],
                "Dedup n":        r["dedup_n"],
                "Raw avg P&L":    f"{r['raw_avg_pnl']:+.2f}%"   if r["raw_avg_pnl"]   is not None else "—",
                "Dedup avg P&L":  f"{r['dedup_avg_pnl']:+.2f}%" if r["dedup_avg_pnl"] is not None else "—",
                "Δ":              f"{((r['dedup_avg_pnl'] or 0) - (r['raw_avg_pnl'] or 0)):+.2f}%",
                "Dedup rule":     r["dedup_rule"],
            } for r in dd["by_event_type"]]
            st.dataframe(pd.DataFrame(et_dd_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 8. R-Multiple Analysis ────────────────────────────────────────────
    st.subheader("📐 8 — R-Multiple Analysis (P&L in units of ATR risk)")
    st.caption("R = |entry − stop|. An R-multiple > 0 means the trade made money relative to the risk taken. HIT_TARGET should be ~+2R; HIT_STOP should be ~−1R.")
    rm = r_multiple_analysis(eval_conn)

    if not rm or rm.get("n_with_stop", 0) == 0:
        st.info("No resolved signals with stop_price data yet.")
    else:
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Signals with stop", rm["n_with_stop"])
        rm2.metric("Avg R-multiple",    f"{rm['avg_r_multiple']:+.3f}R" if rm["avg_r_multiple"] is not None else "—")
        rm3.metric("Median R-multiple", f"{rm['median_r_multiple']:+.3f}R" if rm["median_r_multiple"] is not None else "—")
        rm4.metric("Expectancy",        f"{rm['expectancy']:+.3f}R" if rm["expectancy"] is not None else "—",
                   help="Expected R per trade = win_rate × avg_win_R + loss_rate × avg_loss_R")

        col_dist, col_exit = st.columns(2)
        with col_dist:
            st.markdown("**R-multiple distribution**")
            dist_rows = [{"Bucket": d["bucket"], "Count": d["count"]} for d in rm["distribution"]]
            st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)
        with col_exit:
            st.markdown("**By exit type**")
            exit_rows = [
                {"Exit": exit_type,
                 "n": v["n"],
                 "Avg R": f"{v['avg_r']:+.3f}R",
                 "Median R": f"{v['median_r']:+.3f}R"}
                for exit_type, v in rm["by_exit"].items()
            ]
            st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── 9. Win-Rate Confidence Intervals ──────────────────────────────────
    st.subheader("🎲 9 — Win-Rate Confidence Intervals (Wilson 95% CI)")
    st.caption("A 70% win rate on 5 signals is noise. CI width ≤ 20pp = reliable enough to act on.")
    ci_rows_data = win_rate_confidence_intervals(eval_conn)

    if not ci_rows_data:
        st.info("No resolved outcomes for confidence interval analysis.")
    else:
        ci_table_rows = []
        for r in ci_rows_data:
            reliable_tag = "✅" if r["reliable"] else "⚠️"
            ci_table_rows.append({
                "Event":      r["event_type"],
                "n":          r["n"],
                "Wins":       r["wins"],
                "Win rate":   f"{r['win_rate_pct']:.1f}%" if r["win_rate_pct"] is not None else "—",
                "95% CI":     f"[{r['ci_low']:.1f}%, {r['ci_high']:.1f}%]" if r["ci_low"] is not None else "—",
                "CI width":   f"{r['ci_width']:.1f}pp" if r["ci_width"] is not None else "—",
                "Reliable":   reliable_tag,
            })
        st.dataframe(pd.DataFrame(ci_table_rows), use_container_width=True, hide_index=True)
        st.caption("Reliable = CI width ≤ 20 percentage points. Events marked ⚠️ need more data before drawing conclusions.")

    # ── 10. Component Correlation Report ─────────────────────────────────
    st.divider()
    st.subheader("🔬 10 — Component Correlation (which score layer predicts returns?)")
    st.caption(
        "Pearson r between each score component and three outcome metrics. "
        "**|r| > 0.3** is a meaningful signal at this sample size. "
        "RiskPenalty is subtracted in the final score — a negative corr_t5 means "
        "the penalty correctly flagged losing trades."
    )
    cc = component_correlation_report(eval_conn)

    if not cc or not cc.get("overall"):
        st.info("Not enough resolved outcomes with score components for correlation analysis.")
    else:
        def _fmt_corr(v):
            if v is None:
                return "—"
            bar = "▓▓▓" if abs(v) > 0.3 else ("▒▒" if abs(v) > 0.15 else "░")
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.3f} {bar}"

        def _corr_df(rows):
            return pd.DataFrame([{
                "Component":  r["component"],
                "n":          r["n"],
                "corr t+5":   _fmt_corr(r["corr_t5"]),
                "corr alpha": _fmt_corr(r["corr_alpha"]),
                "corr R-mult":_fmt_corr(r["corr_r"]),
            } for r in rows])

        st.markdown("**Overall**")
        st.dataframe(_corr_df(cc["overall"]), use_container_width=True, hide_index=True)

        if cc.get("by_bucket"):
            st.markdown("**By strategy bucket** (buckets with ≥ 10 signals shown)")
            tabs = st.tabs(list(cc["by_bucket"].keys()))
            for tab, (bucket, rows) in zip(tabs, cc["by_bucket"].items()):
                with tab:
                    st.caption(f"n = {len([r for r in rows if r['n'] > 0])} components, sample size varies per metric")
                    st.dataframe(_corr_df(rows), use_container_width=True, hide_index=True)

    # ── 11. Rescore Comparison Report ────────────────────────────────────
    st.divider()
    st.subheader("🔄 11 — Before / After Rescore Comparison")
    st.caption(
        "Re-applies updated risk scoring logic (event-type-aware RSI, earnings strength bonus, "        "choppy→position-size-only) to historical signals. "        "New scores are hypothetical — outcomes are unchanged. "        "Key question: do new high-score signals win more often?"
    )
    rc = rescore_comparison_report(eval_conn)

    if not rc or not rc.get("summary"):
        st.info("Not enough resolved signals with score components for rescore comparison.")
    else:
        s = rc["summary"]

        # ── Summary metrics ───────────────────────────────────────────────────
        cols = st.columns(5)
        cols[0].metric("Signals", s["n"])
        cols[1].metric("Avg Score", f"{s['old_avg_score']:.1f} → {s['new_avg_score']:.1f}",
                       delta=f"{s['avg_score_delta']:+.2f}")
        cols[2].metric("Avg Risk Penalty", f"{s['old_avg_rp']:.1f} → {s['new_avg_rp']:.1f}",
                       delta=f"{s['old_avg_rp'] - s['new_avg_rp']:+.2f} less")
        cols[3].metric("Upgraded", f"{s['pct_upgraded']:.0f}%")
        cols[4].metric("Downgraded", f"{s['pct_downgraded']:.0f}%")

        with st.expander("Score-bucket win rates & avg P&L (old vs new)", expanded=True):
            bucket_data = [{
                "Bucket":        b["bucket"],
                "Old n":         b["old_n"],
                "New n":         b["new_n"],
                "Old win%":      f"{b['old_win_rate']:.1f}%" if b["old_win_rate"] is not None else "—",
                "New win%":      f"{b['new_win_rate']:.1f}%" if b["new_win_rate"] is not None else "—",
                "Old avg P&L%":  f"{b['old_avg_pnl']:+.2f}%" if b["old_avg_pnl"] is not None else "—",
                "New avg P&L%":  f"{b['new_avg_pnl']:+.2f}%" if b["new_avg_pnl"] is not None else "—",
            } for b in rc["score_buckets"]]
            st.dataframe(pd.DataFrame(bucket_data), use_container_width=True, hide_index=True)

        with st.expander("Score→return correlation (old vs new)"):
            corr = rc["correlation"]
            def _fc(v):
                if v is None: return "—"
                bar = "▓▓▓" if abs(v) > 0.3 else ("▒▒" if abs(v) > 0.15 else "░")
                return f"{v:+.3f} {bar}"
            corr_rows = [
                {"Metric": "corr(score, t5_pnl)",   "Old": _fc(corr["old_corr_t5"]),    "New": _fc(corr["new_corr_t5"])},
                {"Metric": "corr(score, alpha_t5)",  "Old": _fc(corr["old_corr_alpha"]), "New": _fc(corr["new_corr_alpha"])},
                {"Metric": "corr(score, R-multiple)","Old": _fc(corr["old_corr_r"]),    "New": _fc(corr["new_corr_r"])},
            ]
            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

        with st.expander("Tier changes (old → new)"):
            if rc.get("tier_changes"):
                tc_rows = [{"Old tier": t["old_tier"], "New tier": t["new_tier"], "Count": t["count"]}
                           for t in rc["tier_changes"]]
                st.dataframe(pd.DataFrame(tc_rows), use_container_width=True, hide_index=True)

        with st.expander("Biggest upgrades (score increased most)"):
            if rc.get("biggest_upgrades"):
                up_rows = [{
                    "Symbol":    u["symbol"],
                    "Date":      u["signal_date"],
                    "Event":     u["event_type"],
                    "Dir":       u["direction"],
                    "Old→New":   f"{u['old_score']:.1f}→{u['new_score']:.1f}",
                    "Δ":         f"{u['delta']:+.1f}",
                    "Earn str":  f"{u['earn_str']:+.1f}",
                    "RP old→new":f"{u['old_rp']:.1f}→{u['new_rp']:.1f}",
                    "t5 P&L%":   f"{u['t5_pnl']:+.2f}%" if u["t5_pnl"] is not None else "—",
                } for u in rc["biggest_upgrades"]]
                st.dataframe(pd.DataFrame(up_rows), use_container_width=True, hide_index=True)

        with st.expander("Biggest downgrades (score decreased most)"):
            if rc.get("biggest_downgrades"):
                dn_rows = [{
                    "Symbol":    d["symbol"],
                    "Date":      d["signal_date"],
                    "Event":     d["event_type"],
                    "Dir":       d["direction"],
                    "Old→New":   f"{d['old_score']:.1f}→{d['new_score']:.1f}",
                    "Δ":         f"{d['delta']:+.1f}",
                    "Earn str":  f"{d['earn_str']:+.1f}",
                    "RP old→new":f"{d['old_rp']:.1f}→{d['new_rp']:.1f}",
                    "t5 P&L%":   f"{d['t5_pnl']:+.2f}%" if d["t5_pnl"] is not None else "—",
                } for d in rc["biggest_downgrades"]]
                st.dataframe(pd.DataFrame(dn_rows), use_container_width=True, hide_index=True)

    # ── 12. Promoted Signal Quality ──────────────────────────────────────
    st.divider()
    st.subheader("📈 12 — Promoted Signal Quality")
    st.caption(
        "How do signals that gained score in the rescore actually perform? "        "Compare each promotion group against the baseline (all resolved signals). "        "Key question: does a larger score increase predict better outcomes?"
    )
    pq = promoted_signal_quality_report(eval_conn)

    if not pq or not pq.get("groups"):
        st.info("Not enough resolved signals for promoted signal quality analysis.")
    else:
        def _pct(v):
            return f"{v:.1f}%" if v is not None else "—"
        def _f2(v):
            return f"{v:+.2f}%" if v is not None else "—"
        def _r(v):
            return f"{v:+.3f}" if v is not None else "—"

        pq_rows = []
        for g in pq["groups"]:
            pq_rows.append({
                "Group":            g["group"],
                "n":                g["n"],
                "Win rate":         _pct(g["win_rate"]),
                "Avg t5 P&L%":      _f2(g["avg_t5_pnl"]),
                "Avg alpha_t5%":    _f2(g["avg_alpha_t5"]),
                "Avg R-mult":       _r(g["avg_r_multiple"]),
                "Hit target%":      _pct(g["hit_target_rate"]),
                "Hit stop%":        _pct(g["hit_stop_rate"]),
                "T5 exit%":         _pct(g["t5_exit_rate"]),
            })

        st.dataframe(pd.DataFrame(pq_rows), use_container_width=True, hide_index=True)
        st.caption(
            "First row (baseline) = all resolved signals. "            "Subsequent rows are subsets. n < 5 = treat as noise."
        )

    # ── 13. False Upgrade Diagnosis ──────────────────────────────────────
    st.divider()
    st.subheader("🔍 13 — False Upgrade Diagnosis (Δscore ≥ 3, t5 < 0)")
    st.caption(
        "Signals that gained ≥3 points in the rescore but still lost money at t+5. "
        "Goal: find shared characteristics that reveal remaining false-positive patterns."
    )
    fd = false_upgrade_diagnosis_report(eval_conn)

    if not fd or not fd.get("signals"):
        st.info("No false upgrades found (Δscore ≥ 3, t5_pnl < 0). Scoring looks clean!")
    else:
        pat = fd["pattern"]

        # ── Pattern summary ───────────────────────────────────────────────────
        cols = st.columns(5)
        cols[0].metric("False upgrades", pat["n"])
        cols[1].metric("All Δ≥3 signals", pat["all_promoted_n"])
        cols[2].metric("False upgrade rate", f"{pat['false_upgrade_rate']:.0f}%")
        cols[3].metric("% Earnings", f"{pat['pct_earnings']:.0f}%")
        cols[4].metric("% SHORT", f"{pat['pct_short']:.0f}%")

        col2 = st.columns(5)
        col2[0].metric("Avg gap%", f"{pat['avg_gap_pct']:+.1f}%")
        col2[1].metric("Avg RSI", f"{pat['avg_rsi']:.0f}")
        col2[2].metric("Avg ATR%", f"{pat['avg_atr_pct']:.1f}%")
        col2[3].metric("Avg vol ratio", f"{pat['avg_volume_ratio']:.2f}x")
        col2[4].metric("% large gap (≥3%)", f"{pat['pct_large_gap']:.0f}%")

        with st.expander("Pattern breakdown"):
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.markdown("**Event types**")
            for k, v in pat["top_event_types"]:
                pc1.write(f"{k}: {v}")
            pc2.markdown("**Directions**")
            for k, v in pat["top_directions"]:
                pc2.write(f"{k}: {v}")
            pc3.markdown("**Regimes**")
            for k, v in pat["top_regimes"]:
                pc3.write(f"{k}: {v}")
            pc4.markdown("**Strategy buckets**")
            for k, v in pat["top_buckets"]:
                pc4.write(f"{k}: {v}")

        # ── Per-signal table ──────────────────────────────────────────────────
        with st.expander("Per-signal details", expanded=True):
            sig_rows = [{
                "Symbol":       s["symbol"],
                "Date":         s["signal_date"],
                "Dir":          s["direction"],
                "Event":        s["event_type"],
                "Bucket":       s["strategy_bucket"],
                "Regime":       s["regime"],
                "Gap%":         s["gap_pct"],
                "RSI":          s["rsi"],
                "ATR%":         s["atr_pct"],
                "Vol ratio":    s["volume_ratio"],
                "MktConf":      s["market_conf"],
                "RelOpp":       s["relative_opp"],
                "Fresh":        s["freshness"],
                "Old score":    s["old_score"],
                "New score":    s["new_score"],
                "Δ":            s["score_delta"],
                "Old RP":       s["old_risk_penalty"],
                "New RP":       s["new_risk_penalty"],
                "Earn str":     s["earn_strength"],
                "t5 P&L%":      s["t5_pnl"],
                "Exit":         s["paper_exit"],
                "Paper P&L%":   s["paper_pnl"],
                "Upgrade reason": s["reason_for_upgrade"],
            } for s in fd["signals"]]
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    # ── Section 14: Empirical Threshold Backtest ──────────────────────────
    st.divider()
    st.subheader("14. Empirical Threshold Backtest")
    st.caption(
        "Win rate, alpha, and R-multiple at each score threshold. "
        "Uses rescored (new) scores to reflect current scoring logic. "
        "Percentile cuts are relative to all resolved signals. "
        "Key question: where does real edge kick in?"
    )
    tb = empirical_threshold_backtest(eval_conn)

    if not tb:
        st.info("Not enough resolved signals for threshold backtest.")
    else:
        tb_rows = []
        for row in tb:
            n = row.get("n", 0)
            if n == 0:
                tb_rows.append({"Cut": row["label"], "N": 0})
                continue
            tb_rows.append({
                "Cut":            row["label"],
                "N":              n,
                "Win%":           f"{row['win_rate']:.1f}%" if row.get("win_rate") is not None else "—",
                "Avg t5%":        f"{row['avg_t5']:+.2f}%" if row.get("avg_t5") is not None else "—",
                "Avg α":          f"{row['avg_alpha']:+.2f}%" if row.get("avg_alpha") is not None else "—",
                "Avg R":          f"{row['avg_R']:+.2f}" if row.get("avg_R") is not None else "—",
                "HIT_TARGET%":    f"{row['hit_target_rate']:.1f}%" if row.get("hit_target_rate") is not None else "—",
                "HIT_STOP%":      f"{row['hit_stop_rate']:.1f}%" if row.get("hit_stop_rate") is not None else "—",
                "T5_EXIT%":       f"{row['t5_exit_rate']:.1f}%" if row.get("t5_exit_rate") is not None else "—",
                "Worst t5%":      f"{row['worst_t5']:+.2f}%" if row.get("worst_t5") is not None else "—",
                "Avg size mult":  f"{row['avg_pos_mult']:.2f}" if row.get("avg_pos_mult") is not None else "—",
            })
        st.dataframe(pd.DataFrame(tb_rows), use_container_width=True, hide_index=True)

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
