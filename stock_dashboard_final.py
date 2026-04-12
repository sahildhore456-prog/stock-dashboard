import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📊 AI Stock Market Dashboard")
st.markdown("Live Analysis + Smart Recommendations 🚀")

# ---------------- NIFTY 50 (DYNAMIC + SAFE) ----------------
@st.cache_data(ttl=86400)
def get_nifty50():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)

        response = session.get(url, headers=headers, timeout=5)
        data = response.json()

        nifty_dict = {
            item['symbol']: item['symbol'] + ".NS"
            for item in data['data']
        }

        return nifty_dict

    except:
        st.warning("⚠️ Live NSE failed → Using backup")
        return {
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS"
        }

nifty50 = get_nifty50()

# ---------------- SEARCH ----------------
search = st.text_input("🔍 Search Company")

options = list(nifty50.keys()) if search == "" else [
    k for k in nifty50.keys() if search.lower() in k.lower()
]

stock_name = st.selectbox("Select Company", options)
ticker = nifty50[stock_name]

# ---------------- DATE ----------------
start_date = st.date_input("Start Date", datetime(2023,1,1))
end_date = st.date_input("End Date", datetime.today())

if start_date > end_date:
    st.error("Invalid Date Range")
    st.stop()

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=60)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        return data

    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Date" not in data.columns:
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)

    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found")
    st.stop()

# ---------------- INDICATORS ----------------
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/14).mean()
avg_loss = loss.ewm(alpha=1/14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# ---------------- METRICS ----------------
latest = data.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Close", round(latest['Close'],2))
c2.metric("Open", round(latest['Open'],2))
c3.metric("High", round(latest['High'],2))
c4.metric("Low", round(latest['Low'],2))

# ---------------- ADVANCED CHART ----------------
st.subheader("🕯️ Smart Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
))

fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], name='MA20'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name='MA50'))

fig.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# ---------------- RECOMMENDATION ----------------
st.subheader("🤖 AI Recommendation")

rsi = data['RSI'].iloc[-1]

if rsi < 30 and latest['Close'] > latest['MA20']:
    st.success("🟢 STRONG BUY")
elif rsi < 30:
    st.success("🟢 BUY")
elif rsi > 70:
    st.error("🔴 SELL")
else:
    st.info("🟡 HOLD")

# ---------------- ML PREDICTION ----------------
st.subheader("📈 Prediction")

data['Days'] = np.arange(len(data))

model = LinearRegression()
model.fit(data[['Days']], data['Close'])

pred = model.predict([[len(data)+1]])[0]

st.metric("Tomorrow Price", f"₹ {round(pred,2)}")

# ---------------- TRENDING STOCKS ----------------
st.subheader("🔥 Trending Stocks")

@st.cache_data(ttl=300)
def get_trending(nifty_dict):
    results = []

    for name, tick in list(nifty_dict.items())[:15]:
        try:
            df = yf.download(tick, period="5d", progress=False)

            if len(df) < 2:
                continue

            change = ((df['Close'][-1] - df['Close'][-2]) / df['Close'][-2]) * 100

            results.append((name, round(change,2)))
        except:
            continue

    df_res = pd.DataFrame(results, columns=["Stock", "Change %"])
    return df_res.sort_values(by="Change %", ascending=False).head(5)

trending = get_trending(nifty50)

st.dataframe(trending, use_container_width=True)

# ---------------- DOWNLOAD ----------------
st.download_button("📥 Download Data", data.to_csv(index=False))

# ---------------- REFRESH ----------------
if st.sidebar.button("🔄 Refresh"):
    st.rerun()
