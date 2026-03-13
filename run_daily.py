"""
run_daily.py — single command to run the full daily pipeline.

Usage:
    python run_daily.py              # full run
    python run_daily.py --news-only  # only fetch news
    python run_daily.py --price-only # only fetch prices
    python run_daily.py --report     # only regenerate report
"""
import sys, os, argparse
from datetime import datetime, date

# Make sure modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from modules.db             import init_db, get_conn
from modules.news_collector import collect_news
from modules.price_fetcher  import fetch_prices, get_market_regime
from modules.analyzer       import run_analysis
from modules.report_generator import generate_report
from modules.signal_tracker import record_signals, update_outcomes

def header(msg):
    print(f"\n{'─'*50}")
    print(f"  {msg}")
    print(f"{'─'*50}")

def main():
    parser = argparse.ArgumentParser(description="Alpha Engine — Daily Run")
    parser.add_argument("--news-only",  action="store_true")
    parser.add_argument("--price-only", action="store_true")
    parser.add_argument("--report",     action="store_true")
    parser.add_argument("--no-open",    action="store_true", help="Don't open HTML report")
    args = parser.parse_args()

    start = datetime.now()
    today = date.today().isoformat()
    print(f"\n🚀 Alpha Engine — {today} {start.strftime('%H:%M')}")

    # 1. Ensure DB exists
    header("DB init")
    init_db()

    news_count   = 0
    price_count  = 0
    candidates   = 0

    # 2. News
    if not args.price_only and not args.report:
        header("Fetching news")
        news_count = collect_news()

    # 3. Prices
    if not args.news_only and not args.report:
        header("Fetching prices")
        price_count = fetch_prices()

    # 4. Market regime
    conn = get_conn()
    regime = get_market_regime(conn)
    conn.close()
    print(f"\n[regime] {regime['regime'].upper()} · SPY {regime.get('spy_change', 0):+.2f}%")

    # 5. Update any pending outcomes from previous runs (needs fresh prices)
    update_outcomes(verbose=True)

    # 6. Analyze
    if not args.news_only and not args.price_only:
        header("Running analysis")
        candidates = run_analysis(regime)
        record_signals(regime, today)

    # 7. Report
    header("Generating report")
    report_md = generate_report(regime)

    # 7. Log the run
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO daily_runs
        (run_date, run_at, news_fetched, prices_fetched, candidates_found, market_regime, spy_change_pct)
        VALUES (?,?,?,?,?,?,?)
    """, (today, datetime.now().isoformat(), news_count, price_count,
          candidates, regime.get("regime"), regime.get("spy_change", 0)))
    conn.commit()
    conn.close()

    elapsed = (datetime.now() - start).seconds
    print(f"\n✅ Done in {elapsed}s")
    print(f"   News: {news_count} | Prices: {price_count} | Candidates: {candidates}")

    # 8. Open report
    report_path = os.path.join(os.path.dirname(__file__), "reports", f"report_{today}.html")
    if not args.no_open and os.path.exists(report_path):
        import subprocess, platform
        opener = "open" if platform.system() == "Darwin" else "xdg-open"
        subprocess.Popen([opener, report_path])
        print(f"   📄 Report opened in browser")
    else:
        print(f"   📄 Report saved: {report_path}")

    # Print summary to terminal
    print("\n" + "═"*50)
    # Show top 3 BUY_NOW in terminal
    conn = get_conn()
    top = conn.execute("""
        SELECT symbol, final_score, action, direction, thesis
        FROM trade_candidates
        WHERE run_date=? AND action IN ('ACTIONABLE','WATCHLIST')
        ORDER BY final_score DESC LIMIT 5
    """, (today,)).fetchall()
    conn.close()

    if top:
        print("🎯 TOP PICKS TODAY:\n")
        for c in top:
            icon = "🟢" if c["action"] == "ACTIONABLE" else "🟡"
            print(f"  {icon} {c['symbol']:6s} {c['direction']:5s} score={c['final_score']:.0f}  {c['action']}")
            if c["thesis"]:
                print(f"       {c['thesis'][:80]}...")
            print()
    else:
        print("  No strong candidates today. Stay in cash.")

    print("═"*50)

if __name__ == "__main__":
    main()
