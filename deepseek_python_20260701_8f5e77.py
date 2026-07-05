"""
NSE STOCK ANALYZER — PROFESSIONAL TRADER EDITION
By: Harry | Advanced Multi-Asset Trading Platform
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # Technical Analysis library

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pro Trader Analyzer — Harry",
    page_icon="📈",
    layout="wide"
)

NEWS_API_KEY = "YOUR_NEWS_API_KEY"
ALPHA_VANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# ─────────────────────────────────────────────────────────────────────────────
# ASSET DATABASE
# ─────────────────────────────────────────────────────────────────────────────
ASSET_DATABASE = {
    "HDFCBANK": {"type": "stock", "sector": "Banking", "nse": "HDFCBANK.NS"},
    "ICICIBANK": {"type": "stock", "sector": "Banking", "nse": "ICICIBANK.NS"},
    "SBIN": {"type": "stock", "sector": "Banking", "nse": "SBIN.NS"},
    "KOTAKBANK": {"type": "stock", "sector": "Banking", "nse": "KOTAKBANK.NS"},
    "AXISBANK": {"type": "stock", "sector": "Banking", "nse": "AXISBANK.NS"},
    "BAJFINANCE": {"type": "stock", "sector": "Financials", "nse": "BAJFINANCE.NS"},
    "TCS": {"type": "stock", "sector": "IT", "nse": "TCS.NS"},
    "INFY": {"type": "stock", "sector": "IT", "nse": "INFY.NS"},
    "WIPRO": {"type": "stock", "sector": "IT", "nse": "WIPRO.NS"},
    "HCLTECH": {"type": "stock", "sector": "IT", "nse": "HCLTECH.NS"},
    "RELIANCE": {"type": "stock", "sector": "Energy", "nse": "RELIANCE.NS"},
    "NTPC": {"type": "stock", "sector": "Energy", "nse": "NTPC.NS"},
    "TATAMOTORS": {"type": "stock", "sector": "Auto", "nse": "TATAMOTORS.NS"},
    "MARUTI": {"type": "stock", "sector": "Auto", "nse": "MARUTI.NS"},
    "M&M": {"type": "stock", "sector": "Auto", "nse": "M&M.NS"},
    "TATASTEEL": {"type": "stock", "sector": "Metals", "nse": "TATASTEEL.NS"},
    "HINDALCO": {"type": "stock", "sector": "Metals", "nse": "HINDALCO.NS"},
    "ITC": {"type": "stock", "sector": "FMCG", "nse": "ITC.NS"},
    "HINDUNILVR": {"type": "stock", "sector": "FMCG", "nse": "HINDUNILVR.NS"},
    "SUNPHARMA": {"type": "stock", "sector": "Pharma", "nse": "SUNPHARMA.NS"},
    "BHARTIARTL": {"type": "stock", "sector": "Telecom", "nse": "BHARTIARTL.NS"},
    "GOLD": {"type": "commodity", "symbol": "GC=F", "name": "Gold"},
    "SILVER": {"type": "commodity", "symbol": "SI=F", "name": "Silver"},
    "USDINR": {"type": "forex", "symbol": "USDINR=X", "name": "USD/INR"},
    "NIFTY50": {"type": "index", "symbol": "^NSEI", "name": "NIFTY 50"},
    "BANKNIFTY": {"type": "index", "symbol": "^NSEBANK", "name": "BANK NIFTY"},
}

SECTOR_CORRELATIONS = {
    "Metals": ["GOLD", "SILVER"], "Energy": ["GOLD", "USDINR"],
    "Banking": ["USDINR", "NIFTY50"], "IT": ["USDINR", "NIFTY50"],
    "Auto": ["USDINR"], "FMCG": ["NIFTY50"], "Pharma": ["USDINR"], "Telecom": ["NIFTY50"]
}

# ─────────────────────────────────────────────────────────────────────────────
# CLASSES & SERVICES
# ─────────────────────────────────────────────────────────────────────────────
class NewsAggregator:
    def __init__(self):
        self.api_key = NEWS_API_KEY
    def fetch_news(self, query, days=3):
        if self.api_key == "YOUR_NEWS_API_KEY" or not self.api_key: return []
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {"q": query, "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"), "sortBy": "relevancy", "language": "en", "pageSize": 15, "apiKey": self.api_key}
            response = requests.get(url, params=params, timeout=5)
            return response.json().get("articles", []) if response.status_code == 200 else []
        except: return []
    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles[:15]:
            if article.get("description"):
                blob = TextBlob(article["description"])
                sentiments.append(blob.sentiment.polarity)
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            rating = "Bullish" if avg_sentiment > 0.15 else "Bearish" if avg_sentiment < -0.15 else "Neutral"
            return avg_sentiment, rating
        return 0.0, "Neutral"

class AdvancedTechnicalIndicators:
    @staticmethod
    def calculate_all(df):
        close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
        indicators = {}
        
        # Structural moving averages for trend mapping
        indicators['sma_50'] = ta.trend.sma_indicator(close, window=50)
        indicators['sma_200'] = ta.trend.sma_indicator(close, window=200)
        
        # Standard Core Indicators
        indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        indicators['vwap'] = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        indicators['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        indicators['bollinger_hband'] = ta.volatility.BollingerBands(close).bollinger_hband()
        indicators['bollinger_lband'] = ta.volatility.BollingerBands(close).bollinger_lband()
        indicators['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
        indicators['cci'] = ta.trend.CCIIndicator(high, low, close).cci()
        indicators['rsi'] = ta.momentum.RSIIndicator(close).rsi()
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        
        return indicators

# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL BALANCED QUANT SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedScoring:
    @staticmethod
    def score_asset(tech, info, sentiment_score, price_series):
        """
        Symmetric scoring matrix based on risk/reward variables.
        Max Possible Score: +10 (Strong Market Framework), Min Possible Score: -10 (Severe Warning)
        """
        score = 0
        breakdown = {}
        
        current_price = price_series.iloc[-1]
        
        # 1. INSTITUTIONAL TREND FILTER (Crucial Step: Don't buy crashing assets)
        sma50 = tech['sma_50'].iloc[-1] if not tech['sma_50'].empty else current_price
        sma200 = tech['sma_200'].iloc[-1] if not tech['sma_200'].empty else current_price
        
        trend_score = 0
        if current_price > sma50 and sma50 > sma200:
            trend_score += 3  # Clear Bull Market structure
        elif current_price < sma50 and sma50 < sma200:
            trend_score -= 3  # Bear Market down-spiral
        elif current_price < sma200:
            trend_score -= 2  # Trading beneath macro baseline
            
        score += trend_score
        breakdown["Trend Structure Matrix"] = f"{trend_score:+} pts"

        # 2. MOMENTUM SYMMETRY (RSI & CCI)
        rsi_val = tech['rsi'].iloc[-1] if isinstance(tech['rsi'], pd.Series) else 50
        cci_val = tech['cci'].iloc[-1] if isinstance(tech['cci'], pd.Series) else 0
        
        momentum_score = 0
        if rsi_val > 75 or cci_val > 200:
            momentum_score -= 2  # Deep overextended territory
        elif rsi_val < 28 or cci_val < -200:
            # Only reward low RSI if price is holding above macro 200 SMA (Mean reversion)
            momentum_score += 2 if current_price > sma200 else -1 
        elif 50 < rsi_val <= 65:
            momentum_score += 1  # Healthy bullish continuation zone
            
        score += momentum_score
        breakdown["Momentum Parameters"] = f"{momentum_score:+} pts"

        # 3. VOLATILITY & PRICE LOCATION
        bb_high = tech['bollinger_hband'].iloc[-1]
        bb_low = tech['bollinger_lband'].iloc[-1]
        
        volatility_score = 0
        if current_price >= bb_high:
            volatility_score -= 1  # Statistical ceiling hit
        elif current_price <= bb_low:
            volatility_score += 1 if current_price > sma200 else -1
            
        score += volatility_score
        breakdown["Volatility / Deviation Bounds"] = f"{volatility_score:+} pts"

        # 4. VALUATION METRICS (Safeguard against junk or lack of yfinance info data)
        valuation_score = 0
        if isinstance(info, dict) and info:
            pe = info.get('trailingPE')
            roe = info.get('returnOnEquity')
            
            if pe:
                if pe > 45: valuation_score -= 2  # Premium valuation inflation
                elif pe < 18: valuation_score += 1 # Rational pricing
            if roe:
                if roe > 0.18: valuation_score += 2 # Efficient asset utilization
                elif roe < 0.06: valuation_score -= 1
        
        score += valuation_score
        breakdown["Fundamental Valuation Matrix"] = f"{valuation_score:+} pts"

        # 5. ASSYMETRIC SENTIMENT ANALYSIS
        sentiment_points = 0
        if sentiment_score > 0.25: sentiment_points += 1
        elif sentiment_score < -0.25: sentiment_points -= 2  # Market fear has higher weighting
        
        score += sentiment_points
        breakdown["News Sentiment Weight"] = f"{sentiment_points:+} pts"

        # Bound scores strictly between -10 and +10
        final_score = max(-10, min(10, score))
        verdict, color, advice = EnhancedScoring._derive_actionable_verdict(final_score)
        
        return {"total": final_score, "breakdown": breakdown, "verdict": verdict, "color": color, "advice": advice}

    @staticmethod
    def _derive_actionable_verdict(score):
        if score >= 6: return "STRONG BUY", "green", "High probability structural trend matching entry requirements."
        elif score >= 2: return "BUY", "green", "Bullish structure aligned. Risk reward remains acceptable."
        elif score >= -1: return "NEUTRAL / WATCHLIST", "orange", "No definitive edge present. Await breakout or consolidation phase."
        elif score >= -5: return "REDUCE / SHORT EXPOSURE", "red", "Bearish characteristics developing. Protect profits/capital allocations."
        else: return "AVOID / HIGH RISK", "red", "Severe distribution phase or collapsing fundamentals. Do not catch."

# ─────────────────────────────────────────────────────────────────────────────
# CORE WORKFLOW INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🏦 Institutional Quant Trading Terminal")
    st.caption("Advanced Rule-Based Asset Ranking Engine")
    
    if "quick_pick_sym" not in st.session_state:
        st.session_state.quick_pick_sym = ""

    with st.sidebar:
        st.header("🔍 Analysis Matrix Config")
        asset_type = st.selectbox("Asset Class", ["Stock", "Commodity", "Forex", "Index"])
        
        symbol = st.text_input(
            "Symbol Input", 
            value=st.session_state.quick_pick_sym, 
            placeholder="e.g. RELIANCE, TCS, ZOMATO"
        ).upper().strip()
        
        st.subheader("⚡ Quick Blue-Chip Picks")
        quick_cols = st.columns(2)
        quick_symbols = ["RELIANCE", "TCS", "HDFCBANK", "ITC", "GOLD", "USDINR"]
        for i, sym in enumerate(quick_symbols):
            if quick_cols[i % 2].button(sym, key=f"qp_{sym}"):
                st.session_state.quick_pick_sym = sym
                st.rerun()
        
        timeframe = st.selectbox("Historical Lookback Window", ["3M", "6M", "1Y", "2Y"], index=2)
        risk_per_trade = st.slider("Max Account Exposure Model (%)", 0.5, 5.0, 1.5)
        analyze_button = st.button("Execute Core Analysis Run", type="primary")
    
    active_symbol = symbol if symbol else ""
    if analyze_button and active_symbol:
        with st.spinner(f"Parsing market parameters for {active_symbol}..."):
            try:
                asset_data = fetch_market_engine_data(active_symbol, timeframe)
                if asset_data:
                    display_dashboard_metrics(asset_data, risk_per_trade)
            except Exception as e:
                st.error(f"Execution Error: {e}")

def fetch_market_engine_data(symbol, timeframe):
    period_map = {"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
    period = period_map.get(timeframe, "1y")
    
    asset_info = ASSET_DATABASE.get(symbol, {})
    if not asset_info:
        ticker_symbol = symbol if ("=" in symbol or "^" in symbol) else f"{symbol}.NS"
        asset_type, sector = "stock", "Unknown"
    else:
        ticker_symbol = asset_info.get("nse") or asset_info.get("symbol") or symbol
        asset_type, sector = asset_info.get("type"), asset_info.get("sector", "Unknown")
        
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    if hist.empty or len(hist) < 50:
        raise ValueError(f"Insufficient time-series data scraped for {ticker_symbol}. Minimum required: 50 sessions.")
        
    try: info = ticker.info
    except: info = {}
    
    tech = AdvancedTechnicalIndicators.calculate_all(hist)
    return {"symbol": symbol, "hist": hist, "info": info, "technical": tech, "type": asset_type, "sector": sector}

def display_dashboard_metrics(asset_data, risk_per_trade):
    hist = asset_data["hist"]
    tech = asset_data["technical"]
    
    current_price = hist['Close'].iloc[-1]
    prev_price = hist['Close'].iloc[-2]
    pct_change = ((current_price - prev_price) / prev_price) * 100
    
    # Run dynamic sentiment mapping
    news = NewsAggregator()
    articles = news.fetch_news(f"{asset_data['symbol']} stock")
    sentiment_score, sentiment_lbl = news.analyze_sentiment(articles)
    
    # Calculate Institutional Scoring Model
    quant = EnhancedScoring.score_asset(tech, asset_data["info"], sentiment_score, hist['Close'])
    
    # UI Presentation Layout
    st.markdown(f"## {asset_data['symbol']} Framework Assessment Matrix")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot Price", f"₹{current_price:.2f}", f"{pct_change:.2f}%")
    m2.metric("Volatility Index (ATR)", f"{tech['atr'].iloc[-1]:.2f}")
    m3.metric("Macro Trend Alignment (SMA 200)", f"₹{tech['sma_200'].iloc[-1]:.2f}" if pd.notna(tech['sma_200'].iloc[-1]) else "Calculating...")
    m4.metric("Scraping Volatility", f"{hist['Volume'].iloc[-1]:,.0f} shares")
    
    st.markdown("---")
    
    # Quant Engine Card Output
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown(f"### Score Assessment Rank: ` {quant['total']} / 10 `")
        st.markdown(f"### System Verdict: :{quant['color']}[**{quant['verdict']}**]")
        st.info(f"💡 **Execution Directive:** {quant['advice']}")
    with c2:
        st.write("**Quant Factor Attribute Scoring Weight Breakdown:**")
        for key, val in quant["breakdown"].items():
            st.text(f"• {key.ljust(35)} -> {val}")
            
    st.markdown("---")
    
    # Multi-pane execution charts
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="Price Candles"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=tech['sma_50'], name="50 SMA", line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=tech['sma_200'], name="200 SMA", line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=tech['rsi'], name="RSI Line", line=dict(color='magenta', width=1)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
