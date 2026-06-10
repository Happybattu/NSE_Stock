"""
NSE STOCK ANALYZER — Streamlit Version
By: Harry | R2R x Python x Markets
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="NSE Stock Analyzer — Harry",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# TICKER RESOLVER
# ─────────────────────────────────────────────────────────────────────────────
def resolve_ticker(name: str):
    name = name.strip().upper().replace(" ", "")
    candidates = [name + ".NS", name + ".BO", name] if not name.endswith((".NS", ".BO")) else [name]
    for sym in candidates:
        try:
            t = yf.Ticker(sym)
            df = t.history(period="2y", interval="1d", auto_adjust=True)
            if df is not None and len(df) > 50:
                return sym, t, df
        except Exception:
            continue
    raise ValueError(f"Could not fetch data for '{name}'. Check symbol — e.g. RELIANCE, TCS, INFY")

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_technicals(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))

    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist_m = macd - signal

    bb_mid  = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_up   = bb_mid + 2 * bb_std
    bb_dn   = bb_mid - 2 * bb_std
    bb_pct  = (close - bb_dn) / (bb_up - bb_dn)

    low14   = low.rolling(14).min()
    high14  = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    vol_sma20 = vol.rolling(20).mean()
    w52_high  = close.rolling(252).max()
    w52_low   = close.rolling(252).min()

    mom1m = close.pct_change(21)  * 100
    mom3m = close.pct_change(63)  * 100
    mom6m = close.pct_change(126) * 100
    mom1y = close.pct_change(252) * 100

    last = -1
    return {
        "close": round(float(close.iloc[last]), 2),
        "ma20": round(float(ma20.iloc[last]), 2),
        "ma50": round(float(ma50.iloc[last]), 2),
        "ma200": round(float(ma200.iloc[last]), 2),
        "ema9": round(float(ema9.iloc[last]), 2),
        "ema21": round(float(ema21.iloc[last]), 2),
        "ema50": round(float(ema50.iloc[last]), 2),
        "rsi": round(float(rsi.iloc[last]), 2),
        "macd": round(float(macd.iloc[last]), 4),
        "macd_signal": round(float(signal.iloc[last]), 4),
        "macd_hist": round(float(hist_m.iloc[last]), 4),
        "bb_upper": round(float(bb_up.iloc[last]), 2),
        "bb_lower": round(float(bb_dn.iloc[last]), 2),
        "bb_pct": round(float(bb_pct.iloc[last]), 3),
        "stoch_k": round(float(stoch_k.iloc[last]), 2),
        "stoch_d": round(float(stoch_d.iloc[last]), 2),
        "atr": round(float(atr.iloc[last]), 2),
        "vol_today": int(vol.iloc[last]),
        "vol_sma20": int(vol_sma20.iloc[last]),
        "w52_high": round(float(w52_high.iloc[last]), 2),
        "w52_low": round(float(w52_low.iloc[last]), 2),
        "mom1m": round(float(mom1m.iloc[last]), 2),
        "mom3m": round(float(mom3m.iloc[last]), 2),
        "mom6m": round(float(mom6m.iloc[last]), 2),
        "mom1y": round(float(mom1y.iloc[last]), 2),
        "_close_series": close,
        "_high_series": high,
        "_low_series": low,
        "_ma50_series": ma50,
        "_ma200_series": ma200,
        "_close_df": df,
    }

# ─────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS
# ─────────────────────────────────────────────────────────────────────────────
def get_fundamentals(ticker: yf.Ticker) -> dict:
    info = {}
    try:
        raw = ticker.info
        fields = [
            "longName","sector","industry","country","currency",
            "marketCap","enterpriseValue",
            "trailingPE","forwardPE","priceToBook","priceToSalesTrailingTwelveMonths",
            "trailingEps","forwardEps",
            "dividendYield","payoutRatio","dividendRate",
            "profitMargins","operatingMargins","grossMargins","ebitdaMargins",
            "returnOnEquity","returnOnAssets",
            "revenueGrowth","earningsGrowth","earningsQuarterlyGrowth","revenueQuarterlyGrowth",
            "totalRevenue","grossProfits","ebitda","netIncomeToCommon",
            "totalDebt","totalCash","debtToEquity","currentRatio","quickRatio",
            "bookValue","sharesOutstanding","floatShares",
            "beta","fiftyTwoWeekHigh","fiftyTwoWeekLow",
            "averageVolume","averageVolume10days",
            "shortRatio","shortPercentOfFloat",
            "heldPercentInsiders","heldPercentInstitutions",
            "recommendationMean","recommendationKey","numberOfAnalystOpinions",
        ]
        for f in fields:
            info[f] = raw.get(f)
    except Exception as e:
        st.warning(f"Could not load all fundamental data: {e}")

    rev_growth_yoy = None
    try:
        fin = ticker.financials
        if fin is not None and "Total Revenue" in fin.index and fin.shape[1] >= 2:
            rev_curr = fin.loc["Total Revenue"].iloc[0]
            rev_prev = fin.loc["Total Revenue"].iloc[1]
            if rev_prev and rev_prev != 0:
                rev_growth_yoy = round((rev_curr - rev_prev) / abs(rev_prev) * 100, 2)
    except: pass
    info["rev_growth_yoy"] = rev_growth_yoy

    rev_qoq = None
    try:
        qfin = ticker.quarterly_financials
        if qfin is not None and "Total Revenue" in qfin.index and qfin.shape[1] >= 2:
            r0 = qfin.loc["Total Revenue"].iloc[0]
            r1 = qfin.loc["Total Revenue"].iloc[1]
            if r1 and r1 != 0:
                rev_qoq = round((r0 - r1) / abs(r1) * 100, 2)
    except: pass
    info["rev_qoq"] = rev_qoq

    eps_growth_yoy = None
    try:
        fin = ticker.financials
        if fin is not None and "Net Income" in fin.index and fin.shape[1] >= 2:
            ni0 = fin.loc["Net Income"].iloc[0]
            ni1 = fin.loc["Net Income"].iloc[1]
            if ni1 and ni1 != 0:
                eps_growth_yoy = round((ni0 - ni1) / abs(ni1) * 100, 2)
    except: pass
    info["eps_growth_yoy"] = eps_growth_yoy
    return info

# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def score_stock(tech: dict, fund: dict) -> dict:
    signals = {}
    score   = 0

    c    = tech["close"]
    m50  = tech["ma50"]
    m200 = tech["ma200"]
    e9   = tech["ema9"]
    e21  = tech["ema21"]
    rsi  = tech["rsi"]

    if c > m50 * 1.02:
        signals["Price > MA50 (Bullish)"] = (1, f"₹{c} > MA50 ₹{m50}"); score += 1
    elif c < m50 * 0.98:
        signals["Price < MA50 (Bearish)"] = (-1, f"₹{c} < MA50 ₹{m50}"); score -= 1
    else:
        signals["Price ≈ MA50 (Neutral)"] = (0, f"₹{c} ≈ MA50 ₹{m50}")

    if c > m200 * 1.02:
        signals["Price > MA200 (Bull Trend)"] = (1, f"₹{c} > MA200 ₹{m200}"); score += 1
    elif c < m200 * 0.98:
        signals["Price < MA200 (Bear Trend)"] = (-1, f"₹{c} < MA200 ₹{m200}"); score -= 1
    else:
        signals["Price ≈ MA200 (Neutral)"] = (0, "Around long-term avg")

    if m50 > m200:
        gap = round((m50 - m200) / m200 * 100, 2)
        signals["Golden Cross (MA50>MA200)"] = (1, f"Gap: +{gap}%"); score += 1
    else:
        gap = round((m200 - m50) / m200 * 100, 2)
        signals["Death Cross (MA50<MA200)"] = (-1, f"Gap: -{gap}%"); score -= 1

    if e9 > e21:
        signals["EMA9 > EMA21 (Short Bullish)"] = (1, f"{e9:.2f} > {e21:.2f}"); score += 1
    else:
        signals["EMA9 < EMA21 (Short Bearish)"] = (-1, f"{e9:.2f} < {e21:.2f}"); score -= 1

    if rsi < 30:
        signals["RSI Oversold (<30) — Buy Zone"] = (1, f"RSI={rsi}"); score += 2
    elif rsi > 70:
        signals["RSI Overbought (>70) — Caution"] = (-1, f"RSI={rsi}"); score -= 1
    elif 40 <= rsi <= 60:
        signals["RSI Neutral (40-60)"] = (0, f"RSI={rsi}")
    elif 60 < rsi <= 70:
        signals["RSI Bullish (60-70) — Momentum"] = (1, f"RSI={rsi}"); score += 1

    if tech["macd"] > tech["macd_signal"] and tech["macd_hist"] > 0:
        signals["MACD Bullish Crossover"] = (1, f"MACD={tech['macd']:.3f}"); score += 1
    elif tech["macd"] < tech["macd_signal"] and tech["macd_hist"] < 0:
        signals["MACD Bearish Crossover"] = (-1, f"MACD={tech['macd']:.3f}"); score -= 1
    else:
        signals["MACD Mixed"] = (0, f"MACD={tech['macd']:.3f}")

    vol_ratio = tech["vol_today"] / tech["vol_sma20"] if tech["vol_sma20"] > 0 else 1
    if vol_ratio >= 1.5:
        signals["Volume Surge (1.5x avg)"] = (1, f"{vol_ratio:.1f}x avg volume"); score += 1
    elif vol_ratio < 0.5:
        signals["Volume Dry-Up (<0.5x)"] = (-1, f"{vol_ratio:.1f}x avg — weak conviction"); score -= 1
    else:
        signals["Volume Normal"] = (0, f"{vol_ratio:.1f}x avg volume")

    bp = tech["bb_pct"]
    if bp > 0.95:
        signals["Near BB Upper Band — Overbought"] = (-1, f"BB%={bp:.2f}"); score -= 1
    elif bp < 0.1:
        signals["Near BB Lower Band — Oversold"] = (1, f"BB%={bp:.2f}"); score += 1

    w52_pos = (c - tech["w52_low"]) / (tech["w52_high"] - tech["w52_low"]) * 100 if tech["w52_high"] != tech["w52_low"] else 50
    if w52_pos > 90:
        signals["Near 52W High (>90%)"] = (1, f"{w52_pos:.0f}% of 52W range — strong"); score += 1
    elif w52_pos < 20:
        signals["Near 52W Low (<20%)"] = (-1, f"{w52_pos:.0f}% of 52W range"); score -= 1

    if tech["mom3m"] > 10:
        signals["3M Price Momentum >10%"] = (1, f"+{tech['mom3m']:.1f}%"); score += 1
    elif tech["mom3m"] < -10:
        signals["3M Price Momentum <-10%"] = (-1, f"{tech['mom3m']:.1f}%"); score -= 1

    rg = fund.get("rev_growth_yoy") or (fund.get("revenueGrowth") or 0) * 100
    if rg and rg >= 15:
        signals["Revenue Growth ≥15% YoY (Excellent)"] = (2, f"+{rg:.1f}%"); score += 2
    elif rg and rg >= 10:
        signals["Revenue Growth 10-15% YoY (Good)"] = (1, f"+{rg:.1f}%"); score += 1
    elif rg and rg < 0:
        signals["Revenue Declining YoY"] = (-2, f"{rg:.1f}%"); score -= 2
    elif rg:
        signals[f"Revenue Growth <10% ({rg:.1f}%)"] = (0, "Moderate")

    eg = fund.get("eps_growth_yoy") or (fund.get("earningsGrowth") or 0) * 100
    if eg and eg >= 20:
        signals["EPS Growth ≥20% YoY"] = (2, f"+{eg:.1f}%"); score += 2
    elif eg and eg >= 10:
        signals["EPS Growth 10-20% YoY"] = (1, f"+{eg:.1f}%"); score += 1
    elif eg and eg < 0:
        signals["EPS Declining"] = (-2, f"{eg:.1f}%"); score -= 2

    pe = fund.get("trailingPE")
    if pe:
        if pe < 15:
            signals["P/E Attractive (<15)"] = (1, f"P/E={pe:.1f}x"); score += 1
        elif 15 <= pe <= 25:
            signals["P/E Fair (15-25x)"] = (0, f"P/E={pe:.1f}x")
        elif pe > 40:
            signals["P/E Expensive (>40x)"] = (-1, f"P/E={pe:.1f}x — priced for perfection"); score -= 1

    roe = fund.get("returnOnEquity")
    if roe:
        roe_pct = roe * 100
        if roe_pct >= 20:
            signals["ROE ≥20% (High Quality)"] = (1, f"ROE={roe_pct:.1f}%"); score += 1
        elif roe_pct < 10:
            signals["ROE <10% (Weak Returns)"] = (-1, f"ROE={roe_pct:.1f}%"); score -= 1

    pm = fund.get("profitMargins")
    if pm:
        pm_pct = pm * 100
        if pm_pct >= 15:
            signals["Net Margin ≥15% (Excellent)"] = (1, f"{pm_pct:.1f}%"); score += 1
        elif pm_pct < 5:
            signals["Thin Margins (<5%)"] = (-1, f"{pm_pct:.1f}%"); score -= 1

    dte = fund.get("debtToEquity")
    if dte is not None:
        if dte < 30:
            signals["Low Debt/Equity (<30%)"] = (1, f"D/E={dte:.1f}%"); score += 1
        elif dte > 150:
            signals["High Debt/Equity (>150%)"] = (-1, f"D/E={dte:.1f}% — leveraged"); score -= 1

    rec = fund.get("recommendationKey", "")
    if rec in ("strong_buy", "buy"):
        signals["Analysts: Buy/Strong Buy"] = (1, f"{rec.upper()} ({fund.get('numberOfAnalystOpinions',0)} analysts)"); score += 1
    elif rec in ("sell", "strong_sell"):
        signals["Analysts: Sell/Strong Sell"] = (-1, f"{rec.upper()}"); score -= 1

    ins = fund.get("heldPercentInsiders")
    if ins and ins >= 0.3:
        signals["High Insider Holding (≥30%)"] = (1, f"{ins*100:.1f}%"); score += 1

    return {"signals": signals, "total_score": score}

# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def get_support_resistance(tech: dict):
    close = tech["_close_series"].dropna()
    recent = close.tail(60)
    price = tech["close"]
    levels = sorted(set(round(v, 0) for v in [
        recent.rolling(5).min().dropna().iloc[-1],
        recent.rolling(10).min().dropna().iloc[-1],
        recent.rolling(20).min().dropna().iloc[-1],
        recent.rolling(5).max().dropna().iloc[-1],
        recent.rolling(10).max().dropna().iloc[-1],
        recent.rolling(20).max().dropna().iloc[-1],
        tech["ma50"], tech["ma200"],
        tech["bb_upper"], tech["bb_lower"],
    ]))
    supports    = sorted([l for l in levels if l < price], reverse=True)[:3]
    resistances = sorted([l for l in levels if l > price])[:3]
    return supports, resistances

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
def verdict(score: int, tech: dict):
    if score >= 12:   return "🔥 STRONG BUY",   "green",  "POSITIONAL (3–6 months)"
    elif score >= 7:  return "✅ BUY",           "green",  "SWING / POSITIONAL (2–8 weeks)"
    elif score >= 4:  return "👀 WATCHLIST",     "orange", "Wait for confirmation"
    elif score >= 0:  return "⏳ NEUTRAL",       "orange", "No trade recommended"
    elif score >= -5: return "⚠️ CAUTION",       "red",    "Avoid long; short possible"
    else:             return "❌ AVOID",          "red",    "Stay out / Book losses"

# ─────────────────────────────────────────────────────────────────────────────
# RISK PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
def risk_params(tech: dict) -> dict:
    c   = tech["close"]
    atr = tech["atr"]
    sl  = round(c - 2 * atr, 2)
    t1  = round(c + 2 * atr, 2)
    t2  = round(c + 4 * atr, 2)
    t3  = round(c + 6 * atr, 2)
    rr  = round((t2 - c) / (c - sl), 2) if c != sl else 0
    return {
        "stop_loss": sl, "target_1": t1, "target_2": t2, "target_3": t3,
        "rr_ratio": rr, "atr": atr,
        "sl_pct": round((sl - c) / c * 100, 2),
        "t1_pct": round((t1 - c) / c * 100, 2),
        "t2_pct": round((t2 - c) / c * 100, 2),
    }

def fmt_crore(n):
    if n is None: return "N/A"
    if abs(n) >= 1e7: return f"₹{n/1e7:.2f} Cr"
    return f"₹{n:,.0f}"

def fmt_pct(n):
    if n is None: return "N/A"
    return f"{n*100:.2f}%"

def delta_color(val):
    if val is None: return "normal"
    try:
        v = float(str(val).replace("%","").replace("x",""))
        return "normal" if v == 0 else ("normal" if v > 0 else "inverse")
    except: return "normal"

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📈 NSE Stock Analyzer")
st.caption("By Harry | Full Technical + Fundamental Engine")

col_input, col_btn = st.columns([4, 1])
with col_input:
    symbol_input = st.text_input(
        "Enter NSE symbol",
        placeholder="e.g. RELIANCE, TCS, INFY, HDFCBANK",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

st.caption("NSE stocks: enter symbol directly (e.g. RELIANCE). BSE: add .BO (e.g. RELIANCE.BO). US stocks: AAPL, TSLA etc.")

if analyze_btn and symbol_input:
    with st.spinner(f"Fetching data for {symbol_input.upper()}..."):
        try:
            symbol, ticker, hist = resolve_ticker(symbol_input)
            st.success(f"✓ Data loaded → {symbol}  ({len(hist)} trading days)")

            tech       = compute_technicals(hist)
            fund       = get_fundamentals(ticker)
            score_data = score_stock(tech, fund)
            supports, resistances = get_support_resistance(tech)
            c   = tech["close"]
            vd  = verdict(score_data["total_score"], tech)
            rp  = risk_params(tech)
            total = score_data["total_score"]

            # ── VERDICT BANNER ────────────────────────────────────────────────
            st.divider()
            vcol1, vcol2, vcol3 = st.columns(3)
            with vcol1:
                st.metric("Current Price", f"₹{c}")
            with vcol2:
                st.metric("Signal Score", f"{total} / ~±20")
            with vcol3:
                if vd[1] == "green":
                    st.success(f"**{vd[0]}**  —  {vd[2]}")
                elif vd[1] == "orange":
                    st.warning(f"**{vd[0]}**  —  {vd[2]}")
                else:
                    st.error(f"**{vd[0]}**  —  {vd[2]}")

            # ── PRICE CHART ───────────────────────────────────────────────────
            st.divider()
            st.subheader("Price Chart (2 Years)")
            chart_df = hist[["Close"]].copy()
            chart_df.columns = ["Close"]
            st.line_chart(chart_df)

            # ── COMPANY OVERVIEW ──────────────────────────────────────────────
            st.divider()
            st.subheader("Company Overview")
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Name",     fund.get("longName","N/A"))
            oc2.metric("Sector",   fund.get("sector","N/A"))
            oc3.metric("Industry", fund.get("industry","N/A"))
            oc4.metric("Market Cap", fmt_crore(fund.get("marketCap")))

            # ── PRICE SNAPSHOT ────────────────────────────────────────────────
            st.divider()
            st.subheader("Price Snapshot")
            w52h = tech["w52_high"]
            w52l = tech["w52_low"]
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("52W High",  f"₹{w52h}")
            p2.metric("52W Low",   f"₹{w52l}")
            p3.metric("1M Return", f"{tech['mom1m']:.2f}%", delta=f"{tech['mom1m']:.2f}%")
            p4.metric("3M Return", f"{tech['mom3m']:.2f}%", delta=f"{tech['mom3m']:.2f}%")
            p5.metric("6M Return", f"{tech['mom6m']:.2f}%", delta=f"{tech['mom6m']:.2f}%")
            p6.metric("1Y Return", f"{tech['mom1y']:.2f}%", delta=f"{tech['mom1y']:.2f}%")

            # ── MOVING AVERAGES ───────────────────────────────────────────────
            st.divider()
            st.subheader("Moving Averages")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("MA20",  f"₹{tech['ma20']}",  "ABOVE" if c > tech['ma20']  else "BELOW")
            m2.metric("MA50",  f"₹{tech['ma50']}",  "ABOVE" if c > tech['ma50']  else "BELOW")
            m3.metric("MA200", f"₹{tech['ma200']}", "ABOVE" if c > tech['ma200'] else "BELOW")
            m4.metric("EMA9",  f"₹{tech['ema9']}")
            m5.metric("EMA21", f"₹{tech['ema21']}", "EMA9 > EMA21 ✓" if tech['ema9'] > tech['ema21'] else "EMA9 < EMA21 ✗")

            cross_label = "🟢 GOLDEN CROSS (MA50 > MA200)" if tech['ma50'] > tech['ma200'] else "🔴 DEATH CROSS (MA50 < MA200)"
            if tech['ma50'] > tech['ma200']:
                st.success(cross_label)
            else:
                st.error(cross_label)

            # ── OSCILLATORS ───────────────────────────────────────────────────
            st.divider()
            st.subheader("Oscillators & Momentum")
            o1, o2, o3, o4 = st.columns(4)
            rsi_val = tech["rsi"]
            rsi_label = "Oversold — BUY ZONE 🟢" if rsi_val < 30 else ("Overbought — CAUTION 🔴" if rsi_val > 70 else "Neutral 🟡")
            o1.metric("RSI (14)", f"{rsi_val:.2f}", rsi_label)
            o2.metric("MACD", f"{tech['macd']:.4f}", "Bullish ✓" if tech['macd'] > tech['macd_signal'] else "Bearish ✗")
            o3.metric("Stoch %K", f"{tech['stoch_k']:.2f}", "Oversold" if tech['stoch_k'] < 20 else ("Overbought" if tech['stoch_k'] > 80 else "Neutral"))
            o4.metric("BB %B", f"{tech['bb_pct']:.2f}", "Near Top" if tech['bb_pct'] > 0.9 else ("Near Bottom" if tech['bb_pct'] < 0.1 else "Middle"))

            # ── VOLUME ────────────────────────────────────────────────────────
            st.divider()
            st.subheader("Volume Analysis")
            vratio = tech["vol_today"] / tech["vol_sma20"] if tech["vol_sma20"] > 0 else 1
            v1, v2, v3 = st.columns(3)
            v1.metric("Today Volume", f"{tech['vol_today']:,}")
            v2.metric("20D Avg Volume", f"{tech['vol_sma20']:,}")
            v3.metric("Volume Ratio", f"{vratio:.2f}x", "HIGH — Conviction" if vratio > 1.5 else ("LOW — Weak signal" if vratio < 0.5 else "Normal"))

            # ── SUPPORT / RESISTANCE ──────────────────────────────────────────
            st.divider()
            st.subheader("Support & Resistance")
            sr1, sr2 = st.columns(2)
            with sr1:
                st.markdown("**🟢 Support Levels**")
                if supports:
                    for s in supports:
                        st.write(f"₹{s}")
                else:
                    st.write("N/A")
            with sr2:
                st.markdown("**🔴 Resistance Levels**")
                if resistances:
                    for r in resistances:
                        st.write(f"₹{r}")
                else:
                    st.write("N/A")

            # ── FUNDAMENTALS ──────────────────────────────────────────────────
            st.divider()
            st.subheader("Fundamental Analysis")
            rg = fund.get("rev_growth_yoy") or (fund.get("revenueGrowth") or 0) * 100

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Trailing P/E",  f"{fund.get('trailingPE','N/A')}" if fund.get('trailingPE') else "N/A")
            f2.metric("Forward P/E",   f"{fund.get('forwardPE','N/A')}"  if fund.get('forwardPE') else "N/A")
            f3.metric("Price/Book",    f"{fund.get('priceToBook','N/A')}" if fund.get('priceToBook') else "N/A")
            f4.metric("EPS (TTM)",     f"{fund.get('trailingEps','N/A')}")

            f5, f6, f7, f8 = st.columns(4)
            f5.metric("Revenue Growth YoY", f"{rg:.1f}%" if rg else "N/A", delta=f"{rg:.1f}%" if rg else None)
            f6.metric("Profit Margin",      fmt_pct(fund.get("profitMargins")))
            f7.metric("ROE",                fmt_pct(fund.get("returnOnEquity")))
            f8.metric("Debt/Equity",        f"{fund.get('debtToEquity','N/A')}%")

            f9, f10, f11, f12 = st.columns(4)
            f9.metric("Current Ratio",   f"{fund.get('currentRatio','N/A')}")
            f10.metric("Total Cash",     fmt_crore(fund.get("totalCash")))
            f11.metric("Total Debt",     fmt_crore(fund.get("totalDebt")))
            f12.metric("Dividend Yield", fmt_pct(fund.get("dividendYield")))

            st.metric("Analyst Rating", f"{fund.get('recommendationKey','N/A').upper()} ({fund.get('numberOfAnalystOpinions','?')} analysts)")

            # ── TRADE SETUP ───────────────────────────────────────────────────
            st.divider()
            st.subheader("Trade Setup (ATR-Based)")
            t1c, t2c, t3c, t4c, t5c = st.columns(5)
            t1c.metric("Entry (CMP)",  f"₹{c}")
            t2c.metric("Stop Loss",    f"₹{rp['stop_loss']}", f"{rp['sl_pct']:.2f}%")
            t3c.metric("Target 1",     f"₹{rp['target_1']}",  f"+{rp['t1_pct']:.2f}%")
            t4c.metric("Target 2",     f"₹{rp['target_2']}",  f"+{rp['t2_pct']:.2f}%")
            t5c.metric("Risk:Reward",  f"1 : {rp['rr_ratio']}")

            # ── SIGNAL SCORECARD ──────────────────────────────────────────────
            st.divider()
            st.subheader("Signal Scorecard")
            bull = [(k, v) for k, v in score_data["signals"].items() if v[0] > 0]
            bear = [(k, v) for k, v in score_data["signals"].items() if v[0] < 0]
            neut = [(k, v) for k, v in score_data["signals"].items() if v[0] == 0]

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"**🟢 Bullish ({len(bull)})**")
                for k, (s, detail) in bull:
                    weight = "●●" if s == 2 else "●"
                    st.write(f"{weight} {k} — *{detail}*")
            with sc2:
                st.markdown(f"**🔴 Bearish ({len(bear)})**")
                for k, (s, detail) in bear:
                    weight = "●●" if s == -2 else "●"
                    st.write(f"{weight} {k} — *{detail}*")
            with sc3:
                st.markdown(f"**🟡 Neutral ({len(neut)})**")
                for k, (s, detail) in neut:
                    st.write(f"○ {k} — *{detail}*")

            # ── CHECKLIST ─────────────────────────────────────────────────────
            st.divider()
            st.subheader("Quick Checklist")
            rg2 = fund.get("rev_growth_yoy") or (fund.get("revenueGrowth") or 0) * 100
            chk = [
                ("Double-digit Revenue Growth",    rg2 >= 10),
                ("Price > MA50 (Bull momentum)",   c > tech["ma50"]),
                ("Price > MA200 (Long-term bull)", c > tech["ma200"]),
                ("Golden Cross (MA50 > MA200)",    tech["ma50"] > tech["ma200"]),
                ("EMA9 > EMA21 (Short-term bull)", tech["ema9"] > tech["ema21"]),
                ("RSI not overbought (<70)",        tech["rsi"] <= 70),
                ("MACD bullish",                   tech["macd"] > tech["macd_signal"]),
                ("Volume above average",           tech["vol_today"] > tech["vol_sma20"]),
                ("ROE ≥ 15%",                      (fund.get("returnOnEquity") or 0) >= 0.15),
                ("Profit Margins > 10%",           (fund.get("profitMargins") or 0) >= 0.10),
                ("Low Debt (D/E < 100%)",          (fund.get("debtToEquity") or 999) < 100),
                ("Analyst Buy Rating",             fund.get("recommendationKey","") in ("buy","strong_buy")),
            ]
            pass_count = sum(1 for _, v in chk if v)
            st.write(f"**{pass_count}/{len(chk)} criteria met**")

            chk_cols = st.columns(2)
            for i, (name, passed) in enumerate(chk):
                with chk_cols[i % 2]:
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {name}")

            st.divider()
            st.caption("⚠️ Disclaimer: This is algorithmic analysis only. Not SEBI-registered advice. Always do your own research before investing or trading.")

        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            import traceback
            st.code(traceback.format_exc())

elif analyze_btn and not symbol_input:
    st.warning("Please enter a stock symbol first.")
