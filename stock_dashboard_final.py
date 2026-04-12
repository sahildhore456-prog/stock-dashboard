import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Stock Dashboard", layout="wide")

st.title("📊 Smart Stock Market Dashboard")

# ---------------- STOCK LIST (SAFE) ----------------
nifty50 = {
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
    "Bharti Airtel": "BHARTIARTL.NS"
}

# ---------------- SECTOR MAPPING ----------------
sector_map = {
    "Reliance": "Energy",
    "TCS": "IT",
    "Infosys": "IT",
    "Wipro": "IT",
    "HDFC Bank": "Banking",
    "ICICI Bank": "Banking",
    "Axis Bank": "Banking",
    "SBIN": "Banking",
    "ITC": "FMCG",
    "Asian Paints": "FMCG",
    "LT": "Infrastructure",
    "Adani Enterprises": "Energy",
    "Adani Ports": "Logistics",
    "Bajaj Finance": "Finance",
    "Bharti Airtel": "Telecom"
}

# ---------------- SIDEBAR ----------------
stock_name = st.sidebar.selectbox("Select Stock", list(nifty50.keys()))
ticker = nifty50[stock_name]

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=60)
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
    df = df.dropna()
    return df

data = load_data(ticker)

if data.empty:
    st.error("No data found")
    st.stop()

# ---------------- MOVING AVERAGES ----------------
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

data = data.dropna()

# ---------------- CHART ----------------
st.subheader("🕯️ Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="MA20"))
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50"))

st.plotly_chart(fig, use_container_width=True)

# ---------------- AI RECOMMENDATION ----------------
st.subheader("🤖 AI Recommendation")

latest = data.iloc[-1]

if pd.isna(latest['MA20']) or pd.isna(latest['MA50']):
    st.warning("Not enough data")
else:
    close = float(latest['Close'])
    ma20 = float(latest['MA20'])
    ma50 = float(latest['MA50'])

    if close > ma20 and ma20 > ma50:
        rec = "BUY 🟢"
    elif close < ma20 and ma20 < ma50:
        rec = "SELL 🔴"
    else:
        rec = "HOLD 🟡"

    st.info(rec)

# ---------------- PREDICTION ----------------
st.subheader("📈 Prediction")

data['Day'] = np.arange(len(data))
X = data[['Day']]
y = data['Close']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

next_day = np.array([[len(data) + 1]])
pred = model.predict(next_day)

st.metric("Tomorrow Price", f"₹ {round(pred[0],2)}")

# ---------------- TRENDING STOCKS ----------------
st.subheader("🔥 Trending Stocks")

def get_trending():
    results = []

    for name, tick in list(nifty50.items()):
        try:
            df = yf.download(tick, period="5d", progress=False, threads=False)
            if df.empty or len(df) < 2:
                continue

            change = ((df['Close'][-1] - df['Close'][-2]) / df['Close'][-2]) * 100

            results.append((name, change))

        except:
            continue

    df_res = pd.DataFrame(results, columns=["Stock", "Change %"])
    return df_res.sort_values(by="Change %", ascending=False)

trending = get_trending()

if trending.empty:
    st.warning("No trending data")
else:
    st.dataframe(trending.head(5), use_container_width=True)

# ---------------- TOP GAINERS & LOSERS ----------------
st.subheader("📊 Market Movers")

col1, col2 = st.columns(2)

if not trending.empty:
    with col1:
        st.markdown("### 🟢 Top Gainers")
        st.dataframe(trending.head(5), use_container_width=True)

    with col2:
        st.markdown("### 🔴 Top Losers")
        st.dataframe(trending.tail(5), use_container_width=True)

# ---------------- SECTOR PERFORMANCE ----------------
st.subheader("🌍 Sector-wise Trends")

sector_perf = {}

for name, tick in nifty50.items():
    try:
        df = yf.download(tick, period="5d", progress=False, threads=False)
        if df.empty or len(df) < 2:
            continue

        change = ((df['Close'][-1] - df['Close'][-2]) / df['Close'][-2]) * 100

        sector = sector_map.get(name, "Other")

        if sector not in sector_perf:
            sector_perf[sector] = []

        sector_perf[sector].append(change)

    except:
        continue

sector_avg = {k: np.mean(v) for k, v in sector_perf.items()}

sector_df = pd.DataFrame(list(sector_avg.items()), columns=["Sector", "Avg Change %"])

st.bar_chart(sector_df.set_index("Sector"))

# ---------------- DOWNLOAD ----------------
csv = data.to_csv().encode('utf-8')
st.download_button("📥 Download Data", csv, "stock_data.csv", "text/csv")
