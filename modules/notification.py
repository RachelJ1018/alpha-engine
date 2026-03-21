"""
notification.py — Send Alpha Engine daily report via Email and WeChat (PushPlus).

Required env vars:

  Email:
    EMAIL_SENDER      your sending address  (e.g. you@gmail.com)
    EMAIL_PASSWORD    app password (not your login password)
    EMAIL_RECEIVERS   comma-separated list  (defaults to EMAIL_SENDER)
    EMAIL_SMTP_HOST   optional override     (auto-detected from domain)
    EMAIL_SMTP_PORT   optional override     (auto-detected from domain)

  WeChat via PushPlus:
    PUSHPLUS_TOKEN    get free token at https://www.pushplus.plus

At least one channel must be configured; unconfigured channels are skipped silently.

Usage:
    from modules.notification import send_daily_notification
    send_daily_notification(regime, conn)

Test from CLI:
    python -m modules.notification --test
    python -m modules.notification --test --channel email
    python -m modules.notification --test --channel wechat
"""

from __future__ import annotations

import os
import smtplib
import traceback
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Optional


# ── Env helpers ───────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()

def _email_config() -> dict:
    return {
        "sender":    _env("EMAIL_SENDER"),
        "password":  _env("EMAIL_PASSWORD"),
        "receivers": _env("EMAIL_RECEIVERS"),
        "smtp_host": _env("EMAIL_SMTP_HOST"),
        "smtp_port": int(_env("EMAIL_SMTP_PORT", "0") or "0"),
    }

def _pushplus_token() -> str:
    return _env("PUSHPLUS_TOKEN")


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(regime: dict, conn=None) -> dict:
    from modules.db import get_conn
    from modules.report_generator import get_high_conviction_picks
    if conn is None:
        conn = get_conn()

    today = date.today().isoformat()
    regime_label = regime.get("regime", "unknown")

    candidates = conn.execute("""
        SELECT tc.symbol, tc.action, tc.direction, tc.final_score,
               tc.thesis, tc.entry_note, tc.stop_loss_note, tc.target_note,
               tc.strategy_bucket,
               ps.close_price, ps.change_pct, ps.rsi_14
        FROM trade_candidates tc
        LEFT JOIN price_snapshots ps
          ON tc.symbol = ps.symbol AND ps.snapshot_date = ?
        WHERE tc.run_date = ?
          AND tc.action IN ('ACTIONABLE', 'WATCHLIST', 'MONITOR')
        ORDER BY tc.final_score DESC
        LIMIT 10
    """, (today, today)).fetchall()

    return {
        "today":           today,
        "regime":          regime_label.upper(),
        "spy_change":      regime.get("spy_change", 0.0),
        "spy_rsi":         regime.get("spy_rsi", 50),
        "actionable":      [dict(c) for c in candidates if c["action"] == "ACTIONABLE"],
        "watchlist":       [dict(c) for c in candidates if c["action"] == "WATCHLIST"],
        "monitor":         [dict(c) for c in candidates if c["action"] == "MONITOR"],
        "high_conviction": get_high_conviction_picks(conn, today, regime_label),
    }


# ── Formatters ────────────────────────────────────────────────────────────────

def _candidate_line(c: dict) -> str:
    arrow  = "▲" if c["direction"] == "LONG" else "▼"
    price  = f"${c['close_price']:.2f}" if c.get("close_price") else "—"
    change = f" ({c['change_pct']:+.1f}%)" if c.get("change_pct") is not None else ""
    return f"{arrow} {c['symbol']}  {price}{change}  score={c['final_score']:.0f}"


def _format_email(summary: dict) -> tuple[str, str]:
    regime_emoji = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➡", "CHOPPY": "〰"}.get(
        summary["regime"], "❓"
    )
    subject = (
        f"Alpha Engine {summary['today']} · "
        f"{regime_emoji} {summary['regime']} · "
        f"{len(summary['actionable'])} ACTIONABLE"
    )

    # ── High Conviction block ─────────────────────────────────────────────────
    hc_html = ""
    hc_picks = summary.get("high_conviction", [])
    if hc_picks:
        hc_rows = ""
        for p in hc_picks:
            price  = f"${p['close_price']:.2f}" if p.get("close_price") else "—"
            chg    = f"({p['change_pct']:+.1f}%)" if p.get("change_pct") is not None else ""
            bucket = (p.get("strategy_bucket") or "").replace("_", " ")
            thesis = (p.get("thesis") or "")[:140]
            hc_rows += f"""
            <tr>
              <td style="padding:10px 12px;border-bottom:1px solid #fde68a">
                <strong style="font-size:15px">{p['symbol']}</strong>
                <span style="font-size:11px;color:#92400e;margin-left:6px">{bucket}</span>
              </td>
              <td style="padding:10px 12px;border-bottom:1px solid #fde68a;color:#374151">▲ {price} {chg}</td>
              <td style="padding:10px 12px;border-bottom:1px solid #fde68a">
                <span style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">{p['action']}</span>
                <span style="font-size:11px;color:#6b7280;margin-left:4px">EE={p['event_edge_score']:.0f}</span>
              </td>
              <td style="padding:10px 12px;border-bottom:1px solid #fde68a;color:#374151;font-size:13px">{thesis}</td>
            </tr>"""
        hc_html = f"""
  <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:16px;margin-bottom:20px">
    <h3 style="margin:0 0 4px 0;color:#92400e">⭐ High Conviction Picks</h3>
    <p style="margin:0 0 12px 0;font-size:12px;color:#b45309">LONG · EventEdge ≥ 15 · Real catalyst · Non-bear · ≤1 per sector</p>
    <table style="width:100%;border-collapse:collapse">
      <tbody>{hc_rows}</tbody>
    </table>
  </div>"""
    elif summary["regime"] != "BEAR":
        hc_html = """
  <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin-bottom:20px;color:#6b7280;font-size:13px">
    ⭐ No high conviction picks today — no signals pass the eligibility gate.
  </div>"""

    # ── Regular signals table ─────────────────────────────────────────────────
    rows = ""
    for c in summary["actionable"] + summary["watchlist"]:
        color  = "#22c55e" if c["action"] == "ACTIONABLE" else "#f59e0b"
        price  = f"${c['close_price']:.2f}" if c.get("close_price") else "—"
        change = f"({c['change_pct']:+.1f}%)" if c.get("change_pct") is not None else ""
        arrow  = "▲" if c["direction"] == "LONG" else "▼"
        thesis = (c.get("thesis") or "")[:120]
        rows += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb"><strong>{c['symbol']}</strong></td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#6b7280">{arrow} {price} {change}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
            <span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{c['action']}</span>
          </td>
          <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#374151;font-size:13px">{thesis}</td>
        </tr>"""

    if not rows:
        rows = """<tr><td colspan="4" style="padding:16px;color:#6b7280;text-align:center">
            No ACTIONABLE or WATCHLIST signals today — stay in cash.</td></tr>"""

    monitor_str = " · ".join(c["symbol"] for c in summary["monitor"][:8])
    monitor_html = (
        f"<p style='color:#6b7280;font-size:13px;margin-top:16px'>⚪ Monitor: {monitor_str}</p>"
        if monitor_str else ""
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:700px;margin:0 auto;padding:20px;color:#111">
  <h2 style="margin-bottom:4px">📊 Alpha Engine — {summary['today']}</h2>
  <p style="color:#6b7280;margin-top:0">{regime_emoji} <strong>{summary['regime']}</strong> &nbsp;·&nbsp; SPY {summary['spy_change']:+.2f}%</p>
  {hc_html}
  <h3 style="margin-bottom:8px;color:#374151">All Signals</h3>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:#f9fafb">
        <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;font-weight:500">Symbol</th>
        <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;font-weight:500">Price</th>
        <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;font-weight:500">Signal</th>
        <th style="padding:8px 12px;text-align:left;font-size:13px;color:#6b7280;font-weight:500">Thesis</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  {monitor_html}
  <p style="color:#9ca3af;font-size:11px;margin-top:24px;border-top:1px solid #e5e7eb;padding-top:12px">
    Alpha Engine · Personal research tool · Not financial advice
  </p>
</body></html>"""

    return subject, html


def _format_wechat(summary: dict) -> tuple[str, str]:
    """Returns (title, markdown_content) for PushPlus."""
    regime_emoji = {"BULL": "📈", "BEAR": "📉", "NEUTRAL": "➡", "CHOPPY": "〰"}.get(
        summary["regime"], "❓"
    )
    lines = [
        f"# 📊 Alpha Engine · {summary['today']}",
        f"{regime_emoji} **{summary['regime']}** · SPY `{summary['spy_change']:+.2f}%`",
        "",
    ]

    # High conviction picks at top
    hc_picks = summary.get("high_conviction", [])
    if hc_picks:
        lines.append("## ⭐ High Conviction")
        lines.append("*LONG · EE≥15 · Real catalyst · Non-bear · ≤1 per sector*")
        for p in hc_picks:
            price  = f"${p['close_price']:.2f}" if p.get("close_price") else "—"
            bucket = (p.get("strategy_bucket") or "").replace("_", " ")
            lines.append(f"- **{p['symbol']}** ▲ {price}  EE={p['event_edge_score']:.0f}  {p['action']}  _{bucket}_")
            if p.get("thesis"):
                lines.append(f"  - {p['thesis'][:120]}")
        lines.append("")
    elif summary["regime"] != "BEAR":
        lines += ["## ⭐ High Conviction", "*No signals pass the gate today.*", ""]

    if summary["actionable"]:
        lines.append("## 🟢 ACTIONABLE")
        for c in summary["actionable"]:
            arrow = "▲" if c["direction"] == "LONG" else "▼"
            price = f"${c['close_price']:.2f}" if c.get("close_price") else "—"
            lines.append(f"- **{c['symbol']}** {arrow} {price} · score={c['final_score']:.0f}")
            if c.get("thesis"):
                lines.append(f"  - {c['thesis'][:120]}")
    else:
        lines.append("## 🟢 No ACTIONABLE signals today")

    if summary["watchlist"]:
        lines.append("")
        lines.append("## 🟡 WATCHLIST")
        for c in summary["watchlist"]:
            arrow = "▲" if c["direction"] == "LONG" else "▼"
            price = f"${c['close_price']:.2f}" if c.get("close_price") else "—"
            lines.append(f"- **{c['symbol']}** {arrow} {price} · score={c['final_score']:.0f}")

    if summary["monitor"]:
        syms = " · ".join(c["symbol"] for c in summary["monitor"][:8])
        lines += ["", f"⚪ Monitor: {syms}"]

    title = (
        f"Alpha Engine {summary['today']} · "
        f"{summary['regime']} · "
        f"{len(summary['actionable'])} ACTIONABLE"
    )
    return title, "\n".join(lines)


# ── Senders ───────────────────────────────────────────────────────────────────

def _detect_smtp(sender: str, host: str, port: int) -> tuple[str, int]:
    if host and port:
        return host, port
    domain = sender.split("@")[-1].lower() if "@" in sender else ""
    return {
        "gmail.com":    ("smtp.gmail.com",      587),
        "qq.com":       ("smtp.qq.com",          587),
        "163.com":      ("smtp.163.com",         465),
        "126.com":      ("smtp.126.com",         465),
        "outlook.com":  ("smtp.office365.com",   587),
        "hotmail.com":  ("smtp.office365.com",   587),
        "yahoo.com":    ("smtp.mail.yahoo.com",  587),
    }.get(domain, ("smtp.gmail.com", 587))


def _send_email(summary: dict) -> bool:
    cfg      = _email_config()
    sender   = cfg["sender"]
    password = cfg["password"]

    if not sender or not password:
        print("[notify] Email: EMAIL_SENDER / EMAIL_PASSWORD not set")
        return False

    subject, html_body = _format_email(summary)
    receivers = [r.strip() for r in cfg["receivers"].split(",") if r.strip()] or [sender]
    host, port = _detect_smtp(sender, cfg["smtp_host"], cfg["smtp_port"])

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = ", ".join(receivers)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        if port == 465:
            import ssl
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
                s.login(sender, password)
                s.sendmail(sender, receivers, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.ehlo(); s.starttls(); s.ehlo()
                s.login(sender, password)
                s.sendmail(sender, receivers, msg.as_string())
        return True
    except Exception as e:
        print(f"[notify] Email error: {e}")
        return False


def _send_wechat(summary: dict) -> bool:
    import requests
    token = _pushplus_token()
    if not token:
        print("[notify] WeChat: PUSHPLUS_TOKEN not set")
        return False

    title, content = _format_wechat(summary)
    try:
        r = requests.post(
            "https://www.pushplus.plus/send",
            json={"token": token, "title": title, "content": content,
                  "template": "markdown", "channel": "wechat"},
            timeout=20,
        )
        d = r.json()
        if d.get("code") == 200:
            return True
        print(f"[notify] WeChat error: {d}")
        return False
    except Exception as e:
        print(f"[notify] WeChat error: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send_daily_notification(regime: dict, conn=None) -> dict[str, bool]:
    """
    Send today's report to all configured channels (email and/or WeChat).
    Returns {channel: success}. Skips channels without env vars configured.
    """
    cfg = _email_config()
    channels = {
        "email":  bool(cfg["sender"] and cfg["password"]),
        "wechat": bool(_pushplus_token()),
    }
    active = [ch for ch, ok in channels.items() if ok]

    if not active:
        print("[notify] No channels configured — skipping.")
        print("[notify] Set EMAIL_SENDER + EMAIL_PASSWORD and/or PUSHPLUS_TOKEN.")
        return {}

    summary = _build_summary(regime, conn)
    print(f"[notify] Sending to: {', '.join(active)}")

    senders = {"email": _send_email, "wechat": _send_wechat}
    results: dict[str, bool] = {}
    for ch in active:
        try:
            ok = senders[ch](summary)
            results[ch] = ok
            print(f"[notify]   {'✓' if ok else '✗'}  {ch}")
        except Exception:
            results[ch] = False
            traceback.print_exc()

    return results


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    action="store_true")
    parser.add_argument("--channel", default="all", choices=["all", "email", "wechat"])
    args = parser.parse_args()

    cfg = _email_config()
    configured = {
        "email":  bool(cfg["sender"] and cfg["password"]),
        "wechat": bool(_pushplus_token()),
    }

    print("\n" + "═" * 40)
    print("  Alpha Engine — Notification test")
    print("═" * 40)
    print("\nConfigured channels:")
    for ch, ok in configured.items():
        print(f"  {'✓' if ok else '✗'}  {ch}")

    if not any(configured.values()):
        print("""
Set at least one:

  Email:
    export EMAIL_SENDER="you@gmail.com"
    export EMAIL_PASSWORD="app-password"
    export EMAIL_RECEIVERS="you@gmail.com"   # optional, defaults to sender

  WeChat (PushPlus):
    export PUSHPLUS_TOKEN="your-token"
    Get token at: https://www.pushplus.plus
""")
        raise SystemExit(1)

    # Minimal test payload
    test_summary = {
        "today":      date.today().isoformat(),
        "regime":     "NEUTRAL",
        "spy_change": -0.56,
        "spy_rsi":    34.3,
        "actionable": [],
        "watchlist": [{
            "symbol": "NVDA", "action": "WATCHLIST", "direction": "SHORT",
            "final_score": 63, "close_price": 180.25, "change_pct": -1.6,
            "rsi_14": 39.3, "strategy_bucket": "event_short",
            "thesis": "NVDA remains below its 20-day MA amid broader tech selloff.",
            "entry_note": "Break below $179.50",
            "stop_loss_note": "Close above $184.95",
            "target_note": "$172 in 3-5 days",
        }],
        "monitor": [
            {"symbol": "TSLA",  "action": "MONITOR", "direction": "SHORT",
             "final_score": 55, "close_price": 391.2, "change_pct": -1.0,
             "rsi_14": 45.4, "thesis": "", "entry_note": "", "stop_loss_note": "",
             "target_note": "", "strategy_bucket": "event_short"},
        ],
    }

    senders = {"email": _send_email, "wechat": _send_wechat}
    to_test = [args.channel] if args.channel != "all" else [ch for ch, ok in configured.items() if ok]

    print(f"\nSending test to: {', '.join(to_test)}\n")
    for ch in to_test:
        if not configured.get(ch):
            print(f"  ✗  {ch} — not configured")
            continue
        ok = senders[ch](test_summary)
        print(f"  {'✓' if ok else '✗'}  {ch}")

    print("\n" + "═" * 40)
