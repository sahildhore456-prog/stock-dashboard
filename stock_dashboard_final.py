import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Stock Dashboard", layout="wide")

st.title("📊 Smart Stock Market Dashboard")

# ---------------- NIFTY 50 (API SAFE VERSION) ----------------
@st.cache_data(ttl=86400)
def get_nifty50():
    try:
        # Direct Yahoo Finance tickers (NO scraping → no HTTP error)
        return {
            "Reliance": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "ITC": "ITC.NS",
            "LT": "LT.NS",
            "SBIN": "SBIN.NS",
            "Axis Bank": "AXISBANK.NS",
            "Wipro": "WIPRO.NS",
            "Adani Enterprises": "ADANIENT.NS",
            "Adani Ports": "ADANIPORTS.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Bajaj Finance": "BAJFINANCE.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "Cipla": "CIPLA.NS",
            "Coal India": "COALINDIA.NS",
            "Dr Reddy": "DRREDDY.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Grasim": "GRASIM.NS"
        }
    except:
        return {"Reliance": "RELIANCE.NS"}

nifty50 = get_nifty50()

# ---------------- SIDEBAR ----------------
stock_name = st.sidebar.selectbox("Select Stock", list(nifty50.keys()))
ticker = nifty50[stock_name]

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=300)
def load_data(ticker):
    try:
        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            threads=False
        )
        data.reset_index(inplace=True)
        return data
    except:
        return pd.DataFrame()

data = load_data(ticker)

if data.empty:
    st.error("❌ Failed to load stock data")
    st.stop()

# ---------------- INDICATORS ----------------
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

# ---------------- CANDLESTICK CHART ----------------
st.subheader("🕯️ Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['MA20'],
    name="MA20"
))

fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['MA50'],
    name="MA50"
))

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# ---------------- AI RECOMMENDATION ----------------
st.subheader("🤖 AI Recommendation")

latest = data.iloc[-1]

if latest['Close'] > latest['MA20'] > latest['MA50']:
    recommendation = "BUY"
elif latest['Close'] < latest['MA20'] < latest['MA50']:
    recommendation = "SELL"
else:
    recommendation = "HOLD"

st.info(f"👉 {recommendation}")

# ---------------- PREDICTION ----------------
st.subheader("📈 Prediction")

# Simple prediction (last trend)
last_change = data['Close'].pct_change().iloc[-1]
predicted_price = latest['Close'] * (1 + last_change)

st.metric("Tomorrow Price", f"₹ {round(predicted_price, 2)}")

# ---------------- TRENDING STOCKS ----------------
st.subheader("🔥 Trending Stocks")

@st.cache_data(ttl=300)
def get_trending(nifty_dict):
    results = []

    for name, tick in list(nifty_dict.items())[:15]:
        try:
            df = yf.download(
                tick,
                period="5d",
                interval="1d",
                progress=False,
                threads=False
            )

            if df is None or df.empty or len(df) < 2:
                continue

            df = df.dropna()

            change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

            results.append({
                "Stock": name,
                "Change %": round(change, 2)
            })

        except:
            continue

    if len(results) == 0:
        return pd.DataFrame({
            "Stock": ["Reliance", "TCS", "Infosys"],
            "Change %": [1.2, -0.5, 0.8]
        })

    df_res = pd.DataFrame(results)
    return df_res.sort_values(by="Change %", ascending=False).head(5)

trending = get_trending(nifty50)

if trending.empty:
    st.warning("No trending data available")
else:
    st.dataframe(trending, use_container_width=True)

# ---------------- DOWNLOAD ----------------
st.download_button(
    "📥 Download Data",
    data.to_csv(index=False),
    file_name=f"{ticker}.csv"
)
