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
            # Using a free API, replace with actual data source
            url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except:
            # Fallback to static data
            return [
                {"date": "2024-01", "value": 5.1},
                {"date": "2024-02", "value": 5.2},
                {"date": "2024-03", "value": 4.9},
                {"date": "2024-04", "value": 4.8},
            ]
    
    def get_gdp_data(self):
        """Fetch GDP growth data"""
        try:
            # Using a free API, replace with actual data source
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
                    # Align data lengths
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
        
        # Get relevant correlated assets for the sector
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
                        "recommendation": self._get_recommendation(corr, sector)
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
        # RSI
        if indicators.get('rsi', 50) < 30: score += 2
        elif indicators.get('rsi', 50) < 50: score += 1
        elif indicators.get('rsi', 50) > 70: score -= 2
        elif indicators.get('rsi', 50) > 60: score -= 1
        
        # ADX trend strength
        if indicators.get('adx', 25) > 25: score += 1
        if indicators.get('adx', 25) > 40: score += 1
        
        # MACD (simplified)
        # Add MACD scoring logic
        
        return max(-10, min(10, score))
    
    @staticmethod
    def _score_fundamentals(fundamentals):
        score = 0
        # PE Ratio
        pe = fundamentals.get('trailingPE', 0)
        if pe and pe < 15: score += 2
        elif pe and pe < 25: score += 1
        elif pe and pe > 40: score -= 2
        
        # ROE
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
        # Inflation
        inflation = macro_data.get('inflation', 0)
        if inflation and inflation < 3: score += 2
        elif inflation and inflation < 5: score += 1
        elif inflation and inflation > 7: score -= 2
        
        # GDP Growth
        gdp = macro_data.get('gdp', 0)
        if gdp and gdp > 7: score += 2
        elif gdp and gdp > 5: score += 1
        elif gdp and gdp < 3: score -= 2
        
        return max(-10, min(10, score))
    
    @staticmethod
    def _get_verdict(score):
        if score >= 25: return "STRONG BUY", "green", "Excellent opportunity"
        elif score >= 15: return "BUY", "green", "Good fundamentals and technicals"
        elif score >= 5: return "WATCHLIST", "orange", "Monitor for entry"
        elif score >= -5: return "NEUTRAL", "orange", "Wait for better setup"
        elif score >= -15: return "CAUTION", "red", "High risk, avoid long"
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
            pnl_pct = (pnl / (pos["entry_price"] * pos["quantity"])) * 100
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
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Analysis Settings")
        
        # Asset type selector
        asset_type = st.selectbox(
            "Asset Class",
            ["Stock", "Commodity", "Forex", "Index", "Portfolio"]
        )
        
        if asset_type != "Portfolio":
            # Symbol input with auto-complete
            symbol = st.text_input(
                "Symbol",
                placeholder="e.g. RELIANCE, GOLD, USDINR"
            )
            
            # Quick picks
            st.subheader("⚡ Quick Picks")
            quick_cols = st.columns(2)
            quick_symbols = ["RELIANCE", "TCS", "HDFCBANK", "GOLD", "SILVER", "USDINR"]
            for i, sym in enumerate(quick_symbols):
                if quick_cols[i % 2].button(sym):
                    symbol = sym
        
        # Analysis timeframe
        timeframe = st.selectbox(
            "Analysis Period",
            ["1M", "3M", "6M", "1Y", "2Y", "5Y"]
        )
        
        # Show news
        show_news = st.checkbox("📰 Show News & Sentiment", value=True)
        
        # Risk management
        st.subheader("⚖️ Risk Management")
        risk_per_trade = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0)
        max_positions = st.number_input("Max positions", 1, 20, 5)
        
        analyze_button = st.button("🚀 Analyze", type="primary")
    
    # Main content
    if analyze_button and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Fetch data
                asset_data = fetch_asset_data(symbol, timeframe)
                
                if asset_data:
                    # Create tabs for different views
                    tabs = st.tabs([
                        "📊 Overview",
                        "📈 Technicals",
                        "📰 News & Sentiment",
                        "🌍 Macro Analysis",
                        "📊 Correlations",
                        "🎯 Trade Setup",
                        "📁 Portfolio"
                    ])
                    
                    # Tab 1: Overview
                    with tabs[0]:
                        display_overview(asset_data)
                    
                    # Tab 2: Technicals
                    with tabs[1]:
                        display_technical_analysis(asset_data)
                    
                    # Tab 3: News & Sentiment
                    with tabs[2]:
                        if show_news:
                            display_news_sentiment(asset_data)
                    
                    # Tab 4: Macro Analysis
                    with tabs[3]:
                        display_macro_analysis(asset_data)
                    
                    # Tab 5: Correlations
                    with tabs[4]:
                        display_correlation_analysis(asset_data)
                    
                    # Tab 6: Trade Setup
                    with tabs[5]:
                        display_trade_setup(asset_data, risk_per_trade)
                    
                    # Tab 7: Portfolio
                    with tabs[6]:
                        display_portfolio_tracker()
            
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.code(str(e))

def fetch_asset_data(symbol, timeframe):
    """Fetch data for any asset type"""
    # Map timeframe to yfinance period
    period_map = {
        "1M": "1mo", "3M": "3mo", "6M": "6mo", 
        "1Y": "1y", "2Y": "2y", "5Y": "5y"
    }
    period = period_map.get(timeframe, "1y")
    
    # Get asset info from database
    asset_info = ASSET_DATABASE.get(symbol, {})
    
    if not asset_info:
        # Try as direct symbol
        ticker_symbol = symbol
        if not symbol.endswith((".NS", ".BO")):
            ticker_symbol = f"{symbol}.NS"
    else:
        ticker_symbol = asset_info.get("nse", symbol)
    
    # Fetch data
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for {symbol}")
    
    # Get fundamentals
    info = ticker.info
    
    # Calculate technical indicators
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
    
    # Key metrics
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
    
    # Price chart with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
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
    
    # Volume bar chart
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
        height=600,
        title_text=f"{asset_data['symbol']} Price Chart",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis(asset_data):
    """Display comprehensive technical analysis"""
    hist = asset_data["hist"]
    tech = asset_data["technical"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Advanced technical chart with multiple indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Price & Bollinger Bands", "RSI", "MACD")
        )
        
        # Price with Bollinger Bands
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'], name="Close"),
            row=1, col=1
        )
        
        if 'bollinger_hband' in tech:
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['bollinger_hband'], name="BB High", line=dict(dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['bollinger_lband'], name="BB Low", line=dict(dash='dash')),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in tech:
            fig.add_trace(
                go.Scatter(x=hist.index, y=tech['rsi'], name="RSI"),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Technical Indicators Summary")
        
        # Create metrics
        metrics = []
        if 'rsi' in tech:
            rsi_value = tech['rsi'].iloc[-1] if isinstance(tech['rsi'], pd.Series) else tech['rsi']
            metrics.append(("RSI", f"{rsi_value:.2f}", "Bullish" if rsi_value > 50 else "Bearish"))
        
        if 'adx' in tech:
            adx_value = tech['adx'].iloc[-1] if isinstance(tech['adx'], pd.Series) else tech['adx']
            metrics.append(("ADX", f"{adx_value:.2f}", "Strong Trend" if adx_value > 25 else "Weak Trend"))
        
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
        
        # Display top news articles
        st.subheader("📰 Top News Articles")
        for article in articles[:10]:
            with st.expander(article.get('title', 'No title')):
                st.write(f"**Source:** {article.get('source', 'Unknown')}")
                st.write(f"**Date:** {article.get('publishedAt', 'Unknown')}")
                st.write(f"**Description:** {article.get('description', 'No description')}")
                if article.get('url'):
                    st.write(f"[Read more]({article['url']})")
    else:
        st.warning("No news articles found for this symbol")

def display_macro_analysis(asset_data):
    """Display macro economic analysis"""
    macro = MacroIndicators()
    
    # Fetch macro data
    inflation_data = macro.get_inflation_data()
    gdp_data = macro.get_gdp_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Inflation Rate")
        if inflation_data:
            inflation_df = pd.DataFrame(inflation_data)
            st.line_chart(inflation_df.set_index('date')['value'])
    
    with col2:
        st.subheader("📊 GDP Growth")
        if gdp_data:
            gdp_df = pd.DataFrame(gdp_data)
            st.bar_chart(gdp_df.set_index('date')['value'])
    
    # Impact analysis
    st.subheader("🌍 Macro Impact on Assets")
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
    
    # Get correlation data
    correlation_data = {}
    
    # Add correlations with commodities
    for asset in ["GOLD", "SILVER", "COPPER", "ALUMINIUM"]:
        try:
            asset_info = ASSET_DATABASE.get(asset, {})
            asset_ticker = yf.Ticker(asset_info.get("symbol", asset))
            asset_hist = asset_ticker.history(period="1y")
            
            if not asset_hist.empty:
                stock_returns = hist['Close'].pct_change()
                asset_returns = asset_hist['Close'].pct_change()
                
                # Align data
                min_len = min(len(stock_returns), len(asset_returns))
                corr = stock_returns[-min_len:].corr(asset_returns[-min_len:])
                correlation_data[asset] = corr
        except:
            pass
    
    # Add correlations with indices
    for index in ["NIFTY50", "SENSEX"]:
        try:
            asset_info = ASSET_DATABASE.get(index, {})
            asset_ticker = yf.Ticker(asset_info.get("symbol", index))
            asset_hist = asset_ticker.history(period="1y")
            
            if not asset_hist.empty:
                stock_returns = hist['Close'].pct_change()
                asset_returns = asset_hist['Close'].pct_change()
                
                min_len = min(len(stock_returns), len(asset_returns))
                corr = stock_returns[-min_len:].corr(asset_returns[-min_len:])
                correlation_data[index] = corr
        except:
            pass
    
    # Display correlations
    if correlation_data:
        st.subheader("📊 Asset Correlations")
        
        # Create correlation matrix
        corr_df = pd.DataFrame([correlation_data]).T
        corr_df.columns = ['Correlation']
        
        # Color code
        def color_correlation(val):
            if val > 0.5: return 'background-color: green; color: white'
            elif val > 0.2: return 'background-color: lightgreen'
            elif val > -0.2: return 'background-color: yellow'
            elif val > -0.5: return 'background-color: orange'
            else: return 'background-color: red; color: white'
        
        st.dataframe(corr_df.style.applymap(color_correlation))
        
        # Impact assessment
        st.subheader("🎯 Impact on Trading")
        sector = asset_data.get("sector", "Unknown")
        
        for asset, corr in correlation_data.items():
            if abs(corr) > 0.3:
                direction = "positively" if corr > 0 else "negatively"
                strength = "strong" if abs(corr) > 0.5 else "moderate"
                st.info(f"📌 {asset} is {strength} correlated ({direction}) with {symbol}. "
                       f"Consider {asset} for hedging or confirmation.")
    else:
        st.warning("Could not calculate correlations")

def display_trade_setup(asset_data, risk_per_trade):
    """Display comprehensive trade setup"""
    hist = asset_data["hist"]
    tech = asset_data["technical"]
    
    current_price = hist['Close'].iloc[-1]
    
    # Calculate ATR
    atr = tech.get('atr', hist['High'] - hist['Low']).iloc[-1] if isinstance(tech.get('atr'), pd.Series) else tech.get('atr', 1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Entry & Exit Levels")
        
        # Entry
        entry_price = st.number_input("Entry Price", value=float(current_price), step=0.05)
        
        # Stop Loss
        stop_loss = st.number_input("Stop Loss", value=float(current_price - 1.5 * atr), step=0.05)
        
        # Targets
        target1 = st.number_input("Target 1", value=float(current_price + 1.5 * atr), step=0.05)
        target2 = st.number_input("Target 2", value=float(current_price + 3 * atr), step=0.05)
        target3 = st.number_input("Target 3", value=float(current_price + 4.5 * atr), step=0.05)
    
    with col2:
        st.subheader("⚖️ Position Sizing")
        
        # Account size
        account_size = st.number_input("Account Size (₹)", value=1000000, step=100000)
        
        # Position size calculation
        risk_amount = account_size * (risk_per_trade / 100)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share > 0:
            position_size = int(risk_amount / risk_per_share)
        else:
            position_size = 0
        
        st.metric("Position Size", f"{position_size} shares")
        st.metric("Total Exposure", f"₹{position_size * entry_price:,.2f}")
        st.metric("Risk per Trade", f"₹{risk_amount:,.2f} ({risk_per_trade}%)")
        
        # Risk-Reward
        reward1 = target1 - entry_price
        reward2 = target2 - entry_price
        reward3 = target3 - entry_price
        
        rr1 = round(reward1 / risk_per_share, 2) if risk_per_share > 0 else 0
        rr2 = round(reward2 / risk_per_share, 2) if risk_per_share > 0 else 0
        rr3 = round(reward3 / risk_per_share, 2) if risk_per_share > 0 else 0
        
        st.metric("Risk:Reward 1", f"1:{rr1}")
        st.metric("Risk:Reward 2", f"1:{rr2}")
        st.metric("Risk:Reward 3", f"1:{rr3}")
    
    with col3:
        st.subheader("📋 Trade Details")
        
        # Potential profit
        profit1 = position_size * (target1 - entry_price)
        profit2 = position_size * (target2 - entry_price)
        profit3 = position_size * (target3 - entry_price)
        
        st.metric("Potential Profit 1", f"₹{profit1:,.2f}")
        st.metric("Potential Profit 2", f"₹{profit2:,.2f}")
        st.metric("Potential Profit 3", f"₹{profit3:,.2f}")
        
        # Win rate needed
        win_rate_needed = 1 / (1 + rr1) if rr1 > 0 else 0
        st.metric("Win Rate Needed", f"{win_rate_needed * 100:.1f}%")

def display_portfolio_tracker():
    """Display portfolio tracking and management"""
    st.subheader("📁 Portfolio Tracker")
    
    # Initialize portfolio tracker
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PortfolioTracker()
    
    # Add position form
    with st.expander("➕ Add Position"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            symbol = st.text_input("Symbol", placeholder="RELIANCE")
        with col2:
            quantity = st.number_input("Quantity", min_value=1, step=1)
        with col3:
            entry_price = st.number_input("Entry Price", min_value=0.0, step=0.05)
        with col4:
            if st.button("Add to Portfolio"):
                st.session_state.portfolio.add_position(symbol, quantity, entry_price)
                st.success(f"Added {symbol} to portfolio")
    
    # Display portfolio
    if st.session_state.portfolio.positions:
        # Update prices
        prices = {}
        for symbol in st.session_state.portfolio.positions.keys():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    prices[symbol] = hist['Close'].iloc[-1]
            except:
                pass
        
        st.session_state.portfolio.update_prices(prices)
        
        # Show summary
        summary, total_pnl = st.session_state.portfolio.get_portfolio_summary()
        
        # Display as dataframe
        df = pd.DataFrame(summary)
        st.dataframe(df)
        
        # Total P&L
        st.metric("📊 Total Portfolio P&L", f"₹{total_pnl:,.2f}")
    else:
        st.info("No positions in portfolio. Add positions to track.")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()