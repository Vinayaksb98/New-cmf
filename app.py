from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
UNIVERSE_FILE = DATA_DIR / "nifty500_symbols.csv"
NSE_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}

st.set_page_config(page_title="Nifty 500 CMF Breakout EOD Scanner", page_icon="📈", layout="wide")
st.title("📈 Nifty 500 High-Probability CMF Breakout EOD Scanner")
st.caption("End-of-day scan | Run after 3:45 PM IST | Yahoo Finance daily OHLCV data")


def normalize_symbols(values) -> list[str]:
    out = []
    for value in values:
        symbol = str(value).strip().upper()
        if not symbol or symbol in {"NAN", "NONE"}:
            continue
        if symbol.endswith(".NS"):
            symbol = symbol[:-3]
        if symbol.replace("&", "").replace("-", "").replace("_", "").isalnum():
            out.append(symbol + ".NS")
    return sorted(set(out))


def parse_constituent_csv(content: str) -> list[str]:
    frame = pd.read_csv(io.StringIO(content))
    symbol_col = next((c for c in frame.columns if str(c).strip().lower() == "symbol"), None)
    if symbol_col is None:
        raise ValueError(f"CSV has no Symbol column. Found: {list(frame.columns)}")
    symbols = normalize_symbols(frame[symbol_col].dropna().tolist())
    if not 450 <= len(symbols) <= 550:
        raise ValueError(f"Expected approximately 500 constituents, received {len(symbols)}")
    return symbols


def read_saved_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        return []
    try:
        saved = pd.read_csv(UNIVERSE_FILE)
        col = "Symbol" if "Symbol" in saved.columns else saved.columns[0]
        return normalize_symbols(saved[col].dropna().tolist())
    except Exception:
        return []


def refresh_universe() -> tuple[list[str], str]:
    errors = []
    for url in NSE_URLS:
        try:
            response = requests.get(url, headers=NSE_HEADERS, timeout=18)
            response.raise_for_status()
            symbols = parse_constituent_csv(response.content.decode("utf-8-sig", errors="replace"))
            pd.DataFrame({"Symbol": [s.removesuffix(".NS") for s in symbols]}).to_csv(UNIVERSE_FILE, index=False)
            return symbols, f"Fresh official constituent CSV: {url}"
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    saved = read_saved_universe()
    if len(saved) >= 450:
        return saved, "Saved local Nifty 500 constituent file (live refresh failed; using last saved copy)"
    raise RuntimeError("Could not obtain a verified Nifty 500 list. " + " | ".join(errors) + ". Upload the official Nifty 500 CSV below; the app will save it for future runs.")


@st.cache_data(ttl=900, show_spinner=False)
def download_batch(tickers: tuple[str, ...]) -> pd.DataFrame:
    return yf.download(
        tickers=list(tickers), period="6mo", interval="1d", group_by="ticker",
        auto_adjust=True, threads=True, progress=False, timeout=35,
    )


def extract_ticker_frame(raw: pd.DataFrame, ticker: str, batch_size: int) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy() if batch_size == 1 else pd.DataFrame()
    for level in range(raw.columns.nlevels):
        if ticker in raw.columns.get_level_values(level):
            try:
                result = raw.xs(ticker, axis=1, level=level, drop_level=True)
                if isinstance(result.columns, pd.MultiIndex):
                    result.columns = result.columns.get_level_values(-1)
                return result.copy()
            except Exception:
                pass
    return pd.DataFrame()


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    spread = (high - low).replace(0, np.nan)
    multiplier = (((close - low) - (high - close)) / spread).fillna(0.0)
    money_volume = multiplier * volume
    volume_sum = volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return money_volume.rolling(period, min_periods=period).sum() / volume_sum


def scan_stocks(tickers: list[str], cfg: dict, progress_callback=None):
    results, failed = [], []
    chunks = [tickers[i:i + 40] for i in range(0, len(tickers), 40)]
    for batch_no, batch in enumerate(chunks, start=1):
        try:
            raw = download_batch(tuple(batch))
        except Exception as exc:
            failed.extend((t, str(exc)) for t in batch)
            if progress_callback:
                progress_callback(batch_no / len(chunks))
            continue
        for ticker in batch:
            try:
                df = extract_ticker_frame(raw, ticker, len(batch))
                needed = ["High", "Low", "Close", "Volume"]
                if df.empty or not all(c in df.columns for c in needed):
                    failed.append((ticker, "No OHLCV data"))
                    continue
                df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
                df = df[(df["Volume"] > 0) & (df["High"] >= df["Low"])]
                if len(df) < 55:
                    failed.append((ticker, "Insufficient history"))
                    continue
                df["EMA_50"] = df["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
                df["CMF_20"] = compute_cmf(df["High"], df["Low"], df["Close"], df["Volume"], 20)
                df["Res_20D"] = df["High"].shift(1).rolling(20, min_periods=20).max()
                df["Prev_20D_Avg_Vol"] = df["Volume"].shift(1).rolling(20, min_periods=20).mean()
                last = df.iloc[-1]
                vals = [last[c] for c in ["Close", "EMA_50", "CMF_20", "Res_20D", "Volume", "Prev_20D_Avg_Vol"]]
                if not np.isfinite(vals).all() or last["Res_20D"] <= 0 or last["Prev_20D_Avg_Vol"] <= 0:
                    failed.append((ticker, "Indicators unavailable"))
                    continue
                cmp_, ema50, cmf20, res20, vol, prev_avg = map(float, vals)
                vol_ratio = vol / prev_avg
                pct_vs_res = (cmp_ / res20 - 1.0) * 100.0
                turnover_cr = cmp_ * vol / 1e7
                c_trend = cmp_ > ema50
                c_cmf_base = cmf20 > 0.05
                c_breakout = cmp_ > res20
                c_vol_base = vol_ratio >= 2.0
                is_aplus = (
                    c_trend and cmf20 >= cfg["cmf"] and vol_ratio >= cfg["vol"]
                    and cfg["break_min"] <= pct_vs_res <= cfg["break_max"]
                    and turnover_cr >= cfg["turnover"]
                )
                is_near = c_trend and cmf20 >= 0.20 and vol_ratio >= 2.0 and -2.5 <= pct_vs_res <= 0.0
                if is_aplus:
                    tier = "A+ High Probability"
                elif c_trend and c_cmf_base and c_breakout and c_vol_base:
                    tier = "Breakout Confirmed (Base)"
                elif is_near:
                    tier = "Near Breakout Watch"
                else:
                    continue
                # Keep the original score concept, but clamp every component to 0..1.
                cmf_component = np.clip(cmf20 / 0.60, 0, 1)
                vol_component = np.clip(vol_ratio / 8.0, 0, 1)
                breakout_component = np.clip(1 - abs(pct_vs_res - 2.5) / 5.0, 0, 1)
                score = round(cmf_component * 45 + vol_component * 35 + breakout_component * 20, 1)
                sl = risk_pct = t1 = t2 = np.nan
                if tier != "Near Breakout Watch":
                    sl = round(res20 * (1 - cfg["sl"] / 100.0), 2)
                    risk_per_share = max(cmp_ - sl, 0.05)
                    risk_pct = round(risk_per_share / cmp_ * 100.0, 2)
                    t1 = round(cmp_ + 2 * risk_per_share, 2)
                    t2 = round(cmp_ + 3 * risk_per_share, 2)
                results.append({
                    "Tier": tier, "Score": score, "Ticker": ticker.removesuffix(".NS"),
                    "Date": df.index[-1].strftime("%Y-%m-%d"), "CMP": round(cmp_, 2),
                    "20D_Res": round(res20, 2), "Breakout_%": round(pct_vs_res, 2),
                    "EMA_50": round(ema50, 2), "CMF_20": round(cmf20, 3),
                    "Vol_Mult": round(vol_ratio, 2), "Turnover_Cr": round(turnover_cr, 1),
                    "Stop_Loss": sl, "Risk_%": risk_pct, "Target_1_(1:2)": t1, "Target_2_(1:3)": t2,
                })
            except Exception as exc:
                failed.append((ticker, f"Processing error: {exc}"))
        if progress_callback:
            progress_callback(batch_no / len(chunks))
        time.sleep(0.15)
    output = pd.DataFrame(results)
    if not output.empty:
        tier_order = {"A+ High Probability": 0, "Breakout Confirmed (Base)": 1, "Near Breakout Watch": 2}
        output["_order"] = output["Tier"].map(tier_order)
        output = output.sort_values(["_order", "Score"], ascending=[True, False]).drop(columns="_order").reset_index(drop=True)
    return output, failed


def send_telegram(text: str, token: str, chat_id: str) -> tuple[bool, str]:
    if not token or not chat_id:
        return False, "Telegram credentials are blank."
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            return False, str(payload)
        return True, "Telegram alert sent."
    except Exception as exc:
        return False, str(exc)


with st.sidebar:
    st.header("⚙️ Scanner Settings")
    cmf_min = st.number_input("A+ minimum CMF(20)", min_value=0.05, max_value=0.80, value=0.25, step=0.01)
    vol_min = st.number_input("A+ volume multiple", min_value=1.0, max_value=10.0, value=2.3, step=0.1)
    break_min = st.number_input("Minimum breakout %", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    break_max = st.number_input("Maximum breakout %", min_value=1.0, max_value=20.0, value=5.5, step=0.1)
    sl_buffer = st.number_input("Stop-loss buffer below resistance %", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    turnover_min = st.number_input("Minimum daily turnover (₹ Cr)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    st.divider()
    st.subheader("Optional Telegram Alerts")
    bot_token = st.text_input("Bot Token", type="password")
    chat_id = st.text_input("Chat ID")
    st.caption("Credentials are used only when you click Send Telegram Alert; they are not saved by this app.")

cfg = {"cmf": cmf_min, "vol": vol_min, "break_min": break_min, "break_max": break_max, "sl": sl_buffer, "turnover": turnover_min}

col_a, col_b = st.columns([1, 2])
with col_a:
    refresh_clicked = st.button("🔄 Refresh Nifty 500 Constituents", use_container_width=True)
with col_b:
    st.caption("The constituent list is saved locally after the first successful download and reused if NSE is temporarily unavailable.")

uploaded = st.file_uploader("Or upload the official Nifty 500 constituent CSV", type=["csv"])
if uploaded is not None:
    try:
        uploaded_symbols = parse_constituent_csv(uploaded.getvalue().decode("utf-8-sig", errors="replace"))
        pd.DataFrame({"Symbol": [s.removesuffix(".NS") for s in uploaded_symbols]}).to_csv(UNIVERSE_FILE, index=False)
        st.success(f"Saved {len(uploaded_symbols)} verified constituents locally.")
    except Exception as exc:
        st.error(f"Could not use uploaded constituent file: {exc}")

if refresh_clicked:
    st.cache_data.clear()
    try:
        with st.spinner("Refreshing official Nifty 500 constituent list..."):
            symbols, source = refresh_universe()
        st.success(f"Loaded {len(symbols)} constituents. {source}")
    except Exception as exc:
        st.error(str(exc))

try:
    # Always attempt an official refresh on a cold run; then reuse the saved local copy.
    if "universe_symbols" not in st.session_state or refresh_clicked:
        symbols, source = refresh_universe()
        st.session_state["universe_symbols"] = symbols
        st.session_state["universe_source"] = source
    else:
        symbols = st.session_state["universe_symbols"]
        source = st.session_state.get("universe_source", "Saved session universe")
    if len(symbols) < 450:
        raise RuntimeError(f"Only {len(symbols)} constituents loaded; refusing to label this as a Nifty 500 scan.")
    st.success(f"Universe ready: {len(symbols)} stocks | {source}")
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.info("For a true EOD scan, run after 3:45 PM IST. Before market close, Yahoo Finance may return an incomplete daily candle and the result can change.")
if st.button("🚀 Run Nifty 500 EOD Scan", type="primary", use_container_width=True):
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute < 45):
        st.warning("Current IST time is before 3:45 PM. The latest daily candle may be incomplete; scan will still run because you requested it.")
    progress = st.progress(0.0, text="Starting scan...")
    with st.spinner(f"Downloading daily data and scanning {len(symbols)} stocks in batches..."):
        results, failures = scan_stocks(symbols, cfg, lambda p: progress.progress(p, text=f"Scanning batches... {int(p*100)}%"))
    progress.empty()
    st.session_state["scan_results"] = results
    st.session_state["scan_failures"] = failures
    st.session_state["scan_time"] = now_ist.strftime("%d-%m-%Y %H:%M IST")

if "scan_results" in st.session_state:
    results = st.session_state["scan_results"]
    failures = st.session_state["scan_failures"]
    if results.empty:
        st.warning("No stocks matched today's configured EOD criteria.")
    else:
        aplus = results[results["Tier"] == "A+ High Probability"]
        base = results[results["Tier"] == "Breakout Confirmed (Base)"]
        near = results[results["Tier"] == "Near Breakout Watch"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks in universe", len(symbols))
        m2.metric("A+ setups", len(aplus))
        m3.metric("Base breakouts", len(base))
        m4.metric("Near-breakout watch", len(near))
        st.caption(f"Scan completed: {st.session_state['scan_time']} | Data failures/skips: {len(failures)}")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download scan results CSV", results.to_csv(index=False).encode("utf-8-sig"), file_name=f"CMF_Nifty500_EOD_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        if not aplus.empty:
            if st.button("📨 Send A+ Telegram Alert"):
                lines = [f"A+ High-Probability Breakouts ({aplus['Date'].iloc[0]})"]
                for _, row in aplus.iterrows():
                    lines.append(f"{row['Ticker']} @ ₹{row['CMP']} (+{row['Breakout_%']}%) | CMF {row['CMF_20']} | Vol {row['Vol_Mult']}x | SL ₹{row['Stop_Loss']} | T1 ₹{row['Target_1_(1:2)']} | T2 ₹{row['Target_2_(1:3)']}")
                ok, msg = send_telegram("\n".join(lines), bot_token, chat_id)
                (st.success if ok else st.error)(msg)
    if failures:
        with st.expander(f"Data unavailable / skipped symbols ({len(failures)})"):
            st.dataframe(pd.DataFrame(failures, columns=["Ticker", "Reason"]), use_container_width=True, hide_index=True)
