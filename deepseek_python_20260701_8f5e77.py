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

# API Keys (Use environment variables in production)
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Get from newsapi.org
ALPHA_VANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# ─────────────────────────────────────────────────────────────────────────────
# ASSET DATABASE
# ─────────────────────────────────────────────────────────────────────────────
ASSET_DATABASE = {
    # Indian Stocks
    "RELIANCE": {"type": "stock", "sector": "Energy", "nse": "RELIANCE.NS"},
    "TCS": {"type": "stock", "sector": "IT", "nse": "TCS.NS"},
    "INFY": {"type": "stock", "sector": "IT", "nse": "INFY.NS"},
    "HDFCBANK": {"type": "stock", "sector": "Banking", "nse": "HDFCBANK.NS"},
    "ICICIBANK": {"type": "stock", "sector": "Banking", "nse": "ICICIBANK.NS"},
    "SBIN": {"type": "stock", "sector": "Banking", "nse": "SBIN.NS"},
    "BHARTIARTL": {"type": "stock", "sector": "Telecom", "nse": "BHARTIARTL.NS"},
    "TATAMOTORS": {"type": "stock", "sector": "Auto", "nse": "TATAMOTORS.NS"},
    "TATASTEEL": {"type": "stock", "sector": "Metals", "nse": "TATASTEEL.NS"},
    "HINDALCO": {"type": "stock", "sector": "Metals", "nse": "HINDALCO.NS"},
    "JSWSTEEL": {"type": "stock", "sector": "Metals", "nse": "JSWSTEEL.NS"},
    "VEDL": {"type": "stock", "sector": "Metals", "nse": "VEDL.NS"},
    "ADANIENT": {"type": "stock", "sector": "Energy", "nse": "ADANIENT.NS"},
    
    # Commodities (Metals)
    "GOLD": {"type": "commodity", "symbol": "GC=F", "name": "Gold"},
    "SILVER": {"type": "commodity", "symbol": "SI=F", "name": "Silver"},
    "COPPER": {"type": "commodity", "symbol": "HG=F", "name": "Copper"},
    "ALUMINIUM": {"type": "commodity", "symbol": "ALI=F", "name": "Aluminium"},
    
    # Currency
    "USDINR": {"type": "forex", "symbol": "USDINR=X", "name": "USD/INR"},
    "EURINR": {"type": "forex", "symbol": "EURINR=X", "name": "EUR/INR"},
    "GBPINR": {"type": "forex", "symbol": "GBPINR=X", "name": "GBP/INR"},
    
    # Indices
    "NIFTY50": {"type": "index", "symbol": "^NSEI", "name": "NIFTY 50"},
    "SENSEX": {"type": "index", "symbol": "^BSESN", "name": "SENSEX"},
    "BANKNIFTY": {"type": "index", "symbol": "^NSEBANK", "name": "BANK NIFTY"},
}

# Sector correlations for impact analysis
SECTOR_CORRELATIONS = {
    "Metals": ["GOLD", "SILVER", "COPPER", "ALUMINIUM"],
    "Energy": ["GOLD", "USDINR"],
    "Banking": ["USDINR", "NIFTY50"],
    "IT": ["USDINR", "NIFTY50"],
    "Auto": ["ALUMINIUM", "COPPER", "USDINR"],
}

# ─────────────────────────────────────────────────────────────────────────────
# NEWS AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────
class NewsAggregator:
    def __init__(self):
        self.api_key = NEWS_API_KEY
        
    def fetch_news(self, query, days=3):
        """Fetch news from multiple sources"""
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "sortBy": "relevancy",
                "language": "en",
                "pageSize": 50,
                "apiKey": self.api_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get("articles", [])
            return []
        except Exception as e:
            st.warning(f"News fetch error: {e}")
            return []
    
    def analyze_sentiment(self, articles):
        """Analyze sentiment of news articles"""
        sentiments = []
        for article in articles[:20]:  # Limit for performance
            if article.get("description"):
                blob = TextBlob(article["description"])
                sentiments.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "sentiment": blob.sentiment.polarity,
                    "subjectivity": blob.sentiment.subjectivity,
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "publishedAt": article.get("publishedAt", "")
                })
        
        # Calculate aggregate sentiment
        if sentiments:
            avg_sentiment = np.mean([s["sentiment"] for s in sentiments])
            sentiment_rating = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
        else:
            avg_sentiment = 0
            sentiment_rating = "No Data"
            
        return sentiments, avg_sentiment, sentiment_rating

# ─────────────────────────────────────────────────────────────────────────────
# MACRO ECONOMIC INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
class MacroIndicators:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_KEY
        
    def get_inflation_data(self):
        """Fetch inflation data (India)"""
        try:
            url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except:
            return [
                {"date": "2024-01", "value": 5.1},
                {"date": "2024-02", "value": 5.2},
                {"date": "2024-03", "value": 4.9},
                {"date": "2024-04", "value": 4.8},
            ]
    
    def get_gdp_data(self):
        """Fetch GDP growth data"""
        try:
            url = f"https://www.alphavantage.co/query?function=GDP&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except:
            return [
                {"date": "2024-Q1", "value": 7.2},
                {"date": "2024-Q2", "value": 6.8},
                {"date": "2024-Q3", "value": 7.0},
            ]

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedTechnicalIndicators:
    @staticmethod
    def calculate_all(df):
        """Calculate comprehensive technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        indicators = {}
        
        # Volume-based indicators
        indicators['obv'] = ta.volume.OnBalanceVolume(close, volume).on_balance_volume()
        indicators['vwap'] = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
        
        # Volatility indicators
        indicators['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        indicators['bollinger_hband'] = ta.volatility.BollingerBands(close).bollinger_hband()
        indicators['bollinger_lband'] = ta.volatility.BollingerBands(close).bollinger_lband()
        
        # Trend indicators
        indicators['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
        indicators['plus_di'] = ta.trend.ADXIndicator(high, low, close).adx_pos()
        indicators['minus_di'] = ta.trend.ADXIndicator(high, low, close).adx_neg()
        
        # Momentum indicators
        indicators['rsi'] = ta.momentum.RSIIndicator(close).rsi()
        indicators['stoch_k'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
        indicators['stoch_d'] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
        indicators['cci'] = ta.momentum.CommodityChannelIndex(high, low, close).commodity_channel_index()
        indicators['williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        
        return indicators

# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
class CorrelationAnalyzer:
    @staticmethod
    def calculate_correlations(stock_returns, asset_data):
        """Calculate correlations between stock and various assets"""
        correlations = {}
        
        for asset_name, asset_series in asset_data.items():
            if asset_name != "stock":  # Don't correlate with itself
                try:
                    min_len = min(len(stock_returns), len(asset_series))
                    corr = np.corrcoef(
                        stock_returns[-min_len:], 
                        asset_series[-min_len:]
                    )[0, 1]
                    correlations[asset_name] = round(corr, 3)
                except:
                    correlations[asset_name] = 0
        
        return correlations
    
    @staticmethod
    def get_impact_assessment(correlations, sector):
        """Assess potential impact based on correlations"""
        impact = []
        relevant_assets = SECTOR_CORRELATIONS.get(sector, [])
        
        for asset in relevant_assets:
            if asset in correlations:
                corr = correlations[asset]
                if abs(corr) > 0.3:
                    direction = "Positive" if corr > 0 else "Negative"
                    strength = "Strong" if abs(corr) > 0.5 else "Moderate"
                    impact.append({
                        "asset": asset,
                        "correlation": corr,
                        "direction": direction,
                        "strength": strength,
                        "recommendation": CorrelationAnalyzer._get_recommendation(corr, sector)  # FIXED: self -> Class Reference
                    })
        
        return impact
    
    @staticmethod
    def _get_recommendation(corr, sector):
        if corr > 0.5:
            return "Hedge with inverse positions"
        elif corr > 0.3:
            return "Monitor for divergence"
        elif corr < -0.3:
            return "Potential diversification benefit"
        else:
            return "Low correlation - independent trade"

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED STOCK SCORING WITH MACRO FACTORS
# ─────────────────────────────────────────────────────────────────────────────
class EnhancedScoring:
    @staticmethod
    def score_with_macro(tech_indicators, fundamentals, news_sentiment, macro_data):
        """Enhanced scoring with macro factors"""
        score = 0
        signals = {}
        
        # Technical Score
        tech_score = EnhancedScoring._score_technical(tech_indicators)
        score += tech_score
        signals["Technical Score"] = (tech_score, f"{tech_score}/10")
        
        # Fundamental Score
        fund_score = EnhancedScoring._score_fundamentals(fundamentals)
        score += fund_score
        signals["Fundamental Score"] = (fund_score, f"{fund_score}/10")
        
        # Sentiment Score
        sent_score = EnhancedScoring._score_sentiment(news_sentiment)
        score += sent_score
        signals["Sentiment Score"] = (sent_score, f"{sent_score}/10")
        
        # Macro Score
        macro_score = EnhancedScoring._score_macro(macro_data)
        score += macro_score
        signals["Macro Score"] = (macro_score, f"{macro_score}/10")
        
        # Final Verdict
        verdict = EnhancedScoring._get_verdict(score)
        
        return {
            "total_score": score,
            "signals": signals,
            "verdict": verdict,
            "components": {
                "technical": tech_score,
                "fundamental": fund_score,
                "sentiment": sent_score,
                "macro": macro_score
            }
        }
    
    @staticmethod
    def _score_technical(indicators):
        score = 0
        # FIXED: Extract scalar value from pandas Series safely
        rsi_series = indicators.get('rsi')
        rsi = rsi_series.iloc[-1] if isinstance(rsi_series, pd.Series) else float(rsi_series or 50)
        
        if rsi < 30: score += 2
        elif rsi < 50: score += 1
        elif rsi > 70: score -= 2
        elif rsi > 60: score -= 1
        
        adx_series = indicators.get('adx')
        adx = adx_series.iloc[-1] if isinstance(adx_series, pd.Series) else float(adx_series or 25)
        if adx > 25: score += 1
        if adx > 40: score += 1
        
        return max(-10, min(10, score))
    
    @staticmethod
    def _score_fundamentals(fundamentals):
        score = 0
        pe = fundamentals.get('trailingPE', 0)
        if pe and pe < 15: score += 2
        elif pe and pe < 25: score += 1
        elif pe and pe > 40: score -= 2
        
        roe = fundamentals.get('returnOnEquity', 0)
        if roe and roe > 0.15: score += 2
        elif roe and roe > 0.10: score += 1
        elif roe and roe < 0.05: score -= 1
        
        return max(-10, min(10, score))
    
    @staticmethod
    def _score_sentiment(sentiment):
        if sentiment > 0.2: return 2
        elif sentiment > 0.05: return 1
        elif sentiment < -0.1: return -1
        elif sentiment < -0.2: return -2
        return 0
    
    @staticmethod
    def _score_macro(macro_data):
        score = 0
        inflation = macro_data.get('inflation', 0)
        if inflation and inflation < 3: score += 2
        elif inflation and inflation < 5: score += 1
        elif inflation and inflation > 7: score -= 2
        
        gdp = macro_data.get('gdp', 0)
        if gdp and gdp > 7: score += 2
        elif gdp and gdp > 5: score += 1
        elif gdp and gdp < 3: score -= 2
        
        return max(-10, min(10, score))
    
    @staticmethod
    def _get_verdict(score):
        if score >= 10: return "STRONG BUY", "green", "Excellent opportunity"
        elif score >= 3: return "BUY", "green", "Good fundamentals and technicals"
        elif score >= 0: return "WATCHLIST", "orange", "Monitor for entry"
        elif score >= -3: return "NEUTRAL", "orange", "Wait for better setup"
        elif score >= -7: return "CAUTION", "red", "High risk, avoid long"
        else: return "AVOID", "red", "Stay out of this trade"

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioTracker:
    def __init__(self):
        self.positions = {}
        
    def add_position(self, symbol, quantity, entry_price):
        self.positions[symbol] = {
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_date": datetime.now(),
            "current_price": entry_price
        }
    
    def update_prices(self, prices):
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol]["current_price"] = price
    
    def get_portfolio_summary(self):
        summary = []
        total_pnl = 0
        for symbol, pos in self.positions.items():
            pnl = (pos["current_price"] - pos["entry_price"]) * pos["quantity"]
            pnl_pct = (pnl / (pos["entry_price"] * pos["quantity"])) * 100 if (pos["entry_price"] * pos["quantity"]) > 0 else 0
            total_pnl += pnl
            summary.append({
                "symbol": symbol,
                "quantity": pos["quantity"],
                "entry": pos["entry_price"],
                "current": pos["current_price"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2)
            })
        return summary, total_pnl

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🏦 Professional Trading Dashboard")
    st.caption("Multi-Asset Analysis | News Sentiment | Macro Indicators")
    
    # Initialize Quick Pick Session State
    if "quick_pick_sym" not in st.session_state:
        st.session_state.quick_pick_sym = "RELIANCE"
        
    # Sidebar
    with st.sidebar:
        st.header("🔍 Analysis Settings")
        
        asset_type = st.selectbox(
            "Asset Class",
            ["Stock", "Commodity", "Forex", "Index", "Portfolio"]
        )
        
        if asset_type != "Portfolio":
            # FIXED: Linked input to session state for seamless Quick Picks integration
            symbol = st.text_input(
                "Symbol",
                value=st.session_state.quick_pick_sym,
                placeholder="e.g. RELIANCE, GOLD, USDINR"
            )
            
            st.subheader("⚡ Quick Picks")
            quick_cols = st.columns(2)
            quick_symbols = ["RELIANCE", "TCS", "HDFCBANK", "GOLD", "SILVER", "USDINR"]
            for i, sym in enumerate(quick_symbols):
                if quick_cols[i % 2].button(sym, key=f"btn_{sym}"):
                    st.session_state.quick_pick_sym = sym
                    st.reraise_st_flow = True  # Forces rerun with new value
                    st.rerun()
        
        timeframe = st.selectbox(
            "Analysis Period",
            ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
            index=3
        )
        
        show_news = st.checkbox("📰 Show News & Sentiment", value=True)
        
        st.subheader("⚖️ Risk Management")
        risk_per_trade = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0)
        max_positions = st.number_input("Max positions", 1, 20, 5)
        
        analyze_button = st.button("🚀 Analyze", type="primary")
    
    # Main content flow Execution
    if (analyze_button or st.session_state.get("quick_pick_sym")) and asset_type != "Portfolio":
        active_symbol = symbol if symbol else st.session_state.quick_pick_sym
        with st.spinner(f"Analyzing {active_symbol}..."):
            try:
                asset_data = fetch_asset_data(active_symbol, timeframe)
                
                if asset_data:
                    tabs = st.tabs([
                        "📊 Overview",
                        "📈 Technicals",
                        "📰 News & Sentiment",
                        "🌍 Macro Analysis",
                        "📊 Correlations",
                        "🎯 Trade Setup",
                        "📁 Portfolio"
                    ])
                    
                    with tabs[0]:
                        display_overview(asset_data)
                    with tabs[1]:
                        display_technical_analysis(asset_data)
                    with tabs[2]:
                        if show_news:
                            display_news_sentiment(asset_data)
                    with tabs[3]:
                        display_macro_analysis(asset_data)
                    with tabs[4]:
                        display_correlation_analysis(asset_data)
                    with tabs[5]:
                        display_trade_setup(asset_data, risk_per_trade)
                    with tabs[6]:
                        display_portfolio_tracker()
            
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.code(str(e))
    elif asset_type == "Portfolio":
        display_portfolio_tracker()

def fetch_asset_data(symbol, timeframe):
    """Fetch data for any asset type"""
    period_map = {
        "1M": "1mo", "3M": "3mo", "6M": "6mo", 
        "1Y": "1y", "2Y": "2y", "5Y": "5y"
    }
    period = period_map.get(timeframe, "1y")
    asset_info = ASSET_DATABASE.get(symbol, {})
    
    if not asset_info:
        ticker_symbol = symbol
        if not symbol.endswith((".NS", ".BO")):
            ticker_symbol = f"{symbol}.NS"
    else:
        ticker_symbol = asset_info.get("nse") or asset_info.get("symbol") or symbol
    
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {symbol} utilizing identifier: {ticker_symbol}")
    
    try:
        info = ticker.info
    except:
        info = {}
        
    tech = AdvancedTechnicalIndicators.calculate_all(hist)
    
    return {
        "symbol": symbol,
        "ticker": ticker,
        "hist": hist,
        "info": info,
        "technical": tech,
        "asset_type": asset_info.get("type", "stock"),
        "sector": asset_info.get("sector", "Unknown")
    }

def display_overview(asset_data):
    """Display asset overview"""
    hist = asset_data["hist"]
    info = asset_data["info"]
    tech = asset_data["technical"]
    
    col1, col2, col3, col4 = st.columns(4)
    current_price = hist['Close'].iloc[-1]
    price_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:.2f}%")
    with col2:
        st.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
    with col3:
        st.metric("52W High", f"₹{hist['High'].max():.2f}")
    with col4:
        st.metric("52W Low", f"₹{hist['Low'].min():.2f}")
        
    st.markdown("---")
    
    # FIXED: Integration of the previously unused scoring engine 
    st.subheader("🎯 Quantitative Engine Verdict")
    mock_macro = {"inflation": 4.8, "gdp": 7.0} # Fallbacks
    scoring_results = EnhancedScoring.score_with_macro(tech, info, 0.15, mock_macro)
    
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        v_title, v_color, v_desc = scoring_results["verdict"]
        st.markdown(f"### Rating: :{v_color}[**{v_title}**]")
        st.caption(f"Strategy Guideline: {v_desc}")
    with sc2:
        cols = st.columns(4)
        for idx, (label, val_tuple) in enumerate(scoring_results["signals"].items()):
            cols[idx % 4].metric(label, val_tuple[1])
            
    st.markdown("---")
    
    # Price chart with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name="Volume",
            marker_color='rgba(0, 100, 200, 0.5)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title_text=f"{asset_data['symbol']} Price Action Matrix",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis(asset_data):
    """Display comprehensive technical analysis"""
    hist = asset_data["hist"]
    tech = asset_data["technical"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price & Bollinger Bands", "RSI Momentum")
        )
        
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'], name="Close", line=dict(color='aqua')),
            row=1, col=1
        )
        
        if 'bollinger_hband' in tech:
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['bollinger_hband'], name="BB High", line=dict(dash='dash', color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['bollinger_lband'], name="BB Low", line=dict(dash='dash', color='orange')),
                row=1, col=1
            )
        
        if 'rsi' in tech:
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['rsi'], name="RSI", line=dict(color='magenta')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=500, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Technical Indicators Summary")
        
        metrics = []
        if 'rsi' in tech:
            rsi_value = tech['rsi'].iloc[-1] if isinstance(tech['rsi'], pd.Series) else tech['rsi']
            metrics.append(("RSI (14)", f"{rsi_value:.2f}", "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"))
        
        if 'adx' in tech:
            adx_value = tech['adx'].iloc[-1] if isinstance(tech['adx'], pd.Series) else tech['adx']
            metrics.append(("ADX Trend Strength", f"{adx_value:.2f}", "Strong Trend" if adx_value > 25 else "Weak/Sideways"))
        
        if 'cci' in tech:
            cci_value = tech['cci'].iloc[-1] if isinstance(tech['cci'], pd.Series) else tech['cci']
            metrics.append(("CCI", f"{cci_value:.2f}", "Overbought" if cci_value > 100 else "Oversold" if cci_value < -100 else "Neutral"))
        
        for name, value, status in metrics:
            st.metric(name, value, status)

def display_news_sentiment(asset_data):
    """Display news and sentiment analysis"""
    symbol = asset_data["symbol"]
    news_agg = NewsAggregator()
    articles = news_agg.fetch_news(f"{symbol} stock India")
    
    if articles:
        sentiments, avg_sentiment, sentiment_rating = news_agg.analyze_sentiment(articles)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Sentiment Score", f"{avg_sentiment:.3f}")
        with col2:
            st.metric("📈 Sentiment Rating", sentiment_rating)
        with col3:
            st.metric("📰 Articles Found", len(articles))
        
        st.subheader("📰 Top News Articles")
        for article in articles[:10]:
            with st.expander(article.get('title', 'No title')):
                st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown') if isinstance(article.get('source'), dict) else article.get('source', 'Unknown')}")
                st.write(f"**Date:** {article.get('publishedAt', 'Unknown')}")
                st.write(f"**Description:** {article.get('description', 'No description')}")
                if article.get('url'):
                    st.write(f"[Read more]({article['url']})")
    else:
        st.warning("No news articles found for this symbol")

def display_macro_analysis(asset_data):
    """Display macro economic analysis"""
    macro = MacroIndicators()
    inflation_data = macro.get_inflation_data()
    gdp_data = macro.get_gdp_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Inflation Rate")
        if inflation_data:
            st.line_chart(pd.DataFrame(inflation_data).set_index('date')['value'])
    with col2:
        st.subheader("📊 GDP Growth")
        if gdp_data:
            st.bar_chart(pd.DataFrame(gdp_data).set_index('date')['value'])
    
    st.subheader("🌍 Macro Impact Matrix")
    sector = asset_data.get("sector", "Unknown")
    
    impact_analysis = {
        "Inflation": "Negative for equities, positive for commodities" if sector in ["Metals", "Energy"] else "Negative for equities, watch for rate hikes",
        "Interest Rates": "Negative for growth stocks, positive for banks" if sector == "Banking" else "Watch for rate sensitive sectors",
        "GDP Growth": "Positive for cyclical stocks, especially auto & metals" if sector in ["Auto", "Metals"] else "Broad positive for markets",
        "Currency (USD/INR)": "Negative for IT, positive for FII flows" if sector == "IT" else "Affects import/export costs"
    }
    for factor, impact in impact_analysis.items():
        st.write(f"**{factor}:** {impact}")

def display_correlation_analysis(asset_data):
    """Display correlation analysis with other assets"""
    hist = asset_data["hist"]
    symbol = asset_data["symbol"]
    correlation_data = {}
    
    for asset, info in ASSET_DATABASE.items():
        if info.get("type") in ["commodity", "index"] and len(correlation_data) < 6:
            try:
                asset_ticker = yf.Ticker(info.get("symbol", asset))
                asset_hist = asset_ticker.history(period="1y")
                if not asset_hist.empty:
                    stock_returns = hist['Close'].pct_change().dropna()
                    asset_returns = asset_hist['Close'].pct_change().dropna()
                    
                    combined = pd.concat([stock_returns, asset_returns], axis=1).dropna()
                    if not combined.empty:
                        corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                        correlation_data[asset] = round(corr, 3)
            except:
                pass
                
    if correlation_data:
        st.subheader("📊 Cross-Asset Correlations")
        corr_df = pd.DataFrame([correlation_data]).T
        corr_df.columns = ['Correlation Value']
        
        def color_correlation(val):
            if val > 0.4: return 'background-color: rgba(0, 128, 0, 0.3)'
            elif val < -0.4: return 'background-color: rgba(128, 0, 0, 0.3)'
            return 'background-color: rgba(128, 128, 128, 0.1)'
            
        st.dataframe(corr_df.style.map(color_correlation)) # FIXED: applymap replaced with map
        
        st.subheader("🎯 Tactical Alignment Impact")
        for asset, corr in correlation_data.items():
            if abs(corr) > 0.25:
                direction = "positively" if corr > 0 else "negatively"
                st.info(f"📌 **{asset}** is {direction} correlated ({corr}) with {symbol}. Consider structural hedges if deviations occur.")
    else:
        st.warning("Insufficient overlapping records to formulate correlation matrices.")

def display_trade_setup(asset_data, risk_per_trade):
    """Display comprehensive trade setup"""
    hist = asset_data["hist"]
    tech = asset_data["technical"]
    current_price = hist['Close'].iloc[-1]
    
    atr_series = tech.get('atr')
    atr = atr_series.iloc[-1] if isinstance(atr_series, pd.Series) else float(atr_series or (current_price * 0.02))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📊 Execution Levels")
        entry_price = st.number_input("Entry Level", value=float(current_price), step=0.05)
        stop_loss = st.number_input("Stop Loss (SL)", value=float(current_price - 1.5 * atr), step=0.05)
        target1 = st.number_input("Take Profit 1", value=float(current_price + 1.5 * atr), step=0.05)
        target2 = st.number_input("Take Profit 2", value=float(current_price + 3.0 * atr), step=0.05)
    
    with col2:
        st.subheader("⚖️ Position Sizing")
        account_size = st.number_input("Capital Pool (₹)", value=1000000, step=50000)
        
        risk_amount = account_size * (risk_per_trade / 100)
        risk_per_share = abs(entry_price - stop_loss)
        position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        st.metric("Calculated Position Size", f"{position_size} Units")
        st.metric("Gross Exposure Value", f"₹{position_size * entry_price:,.2f}")
        st.metric("Hard Risk Exposure", f"₹{risk_amount:,.2f} ({risk_per_trade}%)")
        
    with col3:
        st.subheader("📋 Reward Metrics")
        rr1 = round((target1 - entry_price) / risk_per_share, 2) if risk_per_share > 0 else 0
        rr2 = round((target2 - entry_price) / risk_per_share, 2) if risk_per_share > 0 else 0
        
        st.metric("R:R Ratio (T1)", f"1:{rr1}")
        st.metric("R:R Ratio (T2)", f"1:{rr2}")
        
        p1 = position_size * (target1 - entry_price)
        st.metric("Expected Yield (Target 1)", f"₹{p1:,.2f}")

def display_portfolio_tracker():
    """Display portfolio tracking and management"""
    st.subheader("📁 Portfolio Tracker")
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PortfolioTracker()
        
    with st.expander("➕ Log New Deployment"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sym = st.text_input("Asset Ticker Symbol", placeholder="RELIANCE")
        with col2:
            qty = st.number_input("Allocated Units", min_value=1, value=10, step=1)
        with col3:
            ent = st.number_input("Execution Cost Basis", min_value=0.0, value=100.0, step=1.0)
        if st.button("Commit Allocation"):
            if sym:
                st.session_state.portfolio.add_position(sym.upper(), qty, ent)
                st.success(f"Asset allocation for {sym.upper()} saved.")
                st.rerun()
                
    if st.session_state.portfolio.positions:
        prices = {}
        for sym in st.session_state.portfolio.positions.keys():
            try:
                ticker_sym = ASSET_DATABASE[sym]["nse"] if sym in ASSET_DATABASE else f"{sym}.NS"
                t = yf.Ticker(ticker_sym)
                h = t.history(period="1d")
                if not h.empty:
                    prices[sym] = h['Close'].iloc[-1]
            except:
                prices[sym] = st.session_state.portfolio.positions[sym]["entry_price"]
                
        st.session_state.portfolio.update_prices(prices)
        summary, total_pnl = st.session_state.portfolio.get_portfolio_summary()
        
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.metric("📊 Aggregated Net P&L Portfolio Realized/Unrealized", f"₹{total_pnl:,.2f}")
    else:
        st.info("No active open exposure tracked inside portfolio databases.")

if __name__ == "__main__":
    main()
