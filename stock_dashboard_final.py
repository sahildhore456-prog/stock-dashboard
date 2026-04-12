import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Smart Stock Market Dashboard (Live)")
st.markdown("Real-time NIFTY 50 Analysis 🚀")

# ---------------- FETCH NIFTY 50 ----------------
@st.cache_data
def get_nifty50_list():
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_50"
        tables = pd.read_html(url)

        nifty_table = tables[1]
        nifty_table = nifty_table[['Company Name', 'Symbol']]

        nifty_dict = {
            row['Company Name']: row['Symbol'] + ".NS"
            for _, row in nifty_table.iterrows()
        }

        return nifty_dict

    except:
        return {
            "Reliance": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "Infosys": "INFY.NS"
        }

nifty50 = get_nifty50_list()

# ---------------- SEARCH ----------------
search = st.text_input("🔍 Search Company")

if search == "":
    options = list(nifty50.keys())
else:
    options = [k for k in nifty50.keys() if search.lower() in k.lower()]

stock_name = st.selectbox("Select Company", options)
ticker = nifty50[stock_name]

# ---------------- DATE ----------------
start_date = st.date_input("Start Date", datetime(2023,1,1))
end_date = st.date_input("End Date", datetime.today())

if start_date > end_date:
    st.error("Start date cannot be after End date")
    st.stop()

# ---------------- LOAD DATA ----------------
@st.cache_data(ttl=60)
def load_data(ticker, start, end):
    import time
    data = pd.DataFrame()

    for i in range(3):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False
            )
            if not data.empty:
                break
        except:
            time.sleep(1)

    if data.empty:
        return data

    # Reset index
    data.reset_index(inplace=True)

    # Fix MultiIndex issue (IMPORTANT)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure Date column
    if "Date" not in data.columns:
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)

    return data

data = load_data(ticker, start_date, end_date)

# ---------------- CHECK DATA ----------------
if data.empty:
    st.error("No data found!")
    st.stop()

if 'Close' not in data.columns:
    st.error(f"Column error: {data.columns}")
    st.stop()

# ---------------- METRICS ----------------
latest = data.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Close", round(latest['Close'],2))
col2.metric("Open", round(latest['Open'],2))
col3.metric("High", round(latest['High'],2))
col4.metric("Low", round(latest['Low'],2))

# ---------------- PRICE CHART ----------------
st.subheader("📈 Price Trend")
fig1 = px.line(data, x="Date", y="Close")
st.plotly_chart(fig1, use_container_width=True)

# ---------------- CANDLESTICK ----------------
st.subheader("🕯️ Candlestick Chart")
fig2 = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])
st.plotly_chart(fig2, use_container_width=True)

# ---------------- MOVING AVERAGE ----------------
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

st.subheader("📊 Moving Average")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], name='MA20'))
fig3.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name='MA50'))
st.plotly_chart(fig3, use_container_width=True)

# ---------------- TREND ----------------
if data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
    st.success("📈 Uptrend (Bullish)")
else:
    st.error("📉 Downtrend (Bearish)")

# ---------------- RSI (EMA) ----------------
delta = data['Close'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(alpha=1/14).mean()
avg_loss = loss.ewm(alpha=1/14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

st.subheader("📉 RSI Indicator")
fig4 = px.line(data, x="Date", y="RSI")
st.plotly_chart(fig4, use_container_width=True)

rsi = data['RSI'].iloc[-1]

if rsi < 30:
    st.success("🟢 BUY Signal (Oversold)")
elif rsi > 70:
    st.error("🔴 SELL Signal (Overbought)")
else:
    st.info("🟡 HOLD")

# ---------------- VOLUME ----------------
st.subheader("📊 Volume")
fig5 = px.bar(data, x="Date", y="Volume")
st.plotly_chart(fig5, use_container_width=True)

# ---------------- VOLATILITY ----------------
volatility = data['Close'].pct_change().std()
st.metric("Volatility", round(volatility,4))

# ---------------- ML PREDICTION ----------------
st.subheader("🤖 AI Prediction")

data['Days'] = np.arange(len(data))

X = data[['Days']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

future_day = [[len(data)+1]]
predicted = model.predict(future_day)[0]

st.metric("Predicted Price", f"₹ {round(predicted,2)}")

# ---------------- DOWNLOAD ----------------
st.download_button(
    "📥 Download Data",
    data.to_csv(index=False),
    file_name="stock_data.csv"
)

# ---------------- REFRESH ----------------
st.sidebar.subheader("🔄 Refresh Data")
if st.sidebar.button("Refresh Now"):
    st.rerun()

st.sidebar.info("Click button to update live data")
