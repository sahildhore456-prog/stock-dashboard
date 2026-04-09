import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Smart Stock Market Dashboard (Live)")
st.markdown("Real-time NIFTY 50 Analysis 🚀")

# ---------------- NIFTY 50 LIST ----------------
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
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr Reddy": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hindalco": "HINDALCO.NS",
    "HUL": "HINDUNILVR.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "LTIMindtree": "LTIM.NS",
    "Maruti": "MARUTI.NS",
    "NTPC": "NTPC.NS",
    "Nestle": "NESTLEIND.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "SBI Life": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "BPCL": "BPCL.NS",
    "Britannia": "BRITANNIA.NS"
}

# ---------------- SEARCH + SELECT ----------------
# ---------------- SMART SEARCH ----------------
search = st.text_input("🔍 Search Company (type few letters)")

# If nothing typed → show ALL companies
if search == "":
    options = list(nifty50.keys())
else:
    # Filter automatically while typing
    options = [k for k in nifty50.keys() if search.lower() in k.lower()]

# Dropdown automatically updates
stock_name = st.selectbox("Select Company", options)

ticker = nifty50[stock_name]
# ---------------- DATE RANGE ----------------
start_date = st.date_input("Start Date", datetime(2023,1,1))
end_date = st.date_input("End Date", datetime.today())

# ---------------- LOAD DATA (FAST CACHE) ----------------
# ---------------- LOAD DATA (FAST + RELIABLE CACHE) ----------------
@st.cache_data(ttl=60)
def load_data(ticker, start_date, end_date):

    import time

    data = pd.DataFrame()

    # Retry logic (3 attempts)
    for i in range(3):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                threads=True
            )

            if not data.empty:
                break

        except Exception as e:
            time.sleep(1)

    # If still empty → return
    if data.empty:
        return data

    # Reset index
    data.reset_index(inplace=True)

    # Fix multi-level columns (VERY IMPORTANT)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure Date column exists
    if "Date" not in data.columns:
        data.rename(columns={data.columns[0]: "Date"}, inplace=True)

    return data


# Call function
data = load_data(ticker, start_date, end_date)
# ---------------- CHECK DATA ----------------
if data.empty:
    st.error("No data found!")
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

# ---------------- RSI ----------------
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

st.subheader("📉 RSI Indicator")

fig4 = px.line(data, x="Date", y="RSI")
st.plotly_chart(fig4, use_container_width=True)

# ---------------- BUY / SELL SIGNAL ----------------
st.subheader("🔮 Buy/Sell Signal")

rsi = data['RSI'].iloc[-1]

if rsi < 30:
    st.success("🟢 BUY Signal (Oversold)")
elif rsi > 70:
    st.error("🔴 SELL Signal (Overbought)")
else:
    st.info("🟡 HOLD")

# ---------------- PREDICTION ----------------
st.subheader("📊 Prediction (Demo)")

predicted = float(data['Close'].iloc[-1]) * 1.02

st.metric("Predicted Price", f"₹ {round(predicted, 2)}")

# ---------------- REFRESH BUTTON ----------------
st.sidebar.subheader("🔄 Refresh Data")

if st.sidebar.button("Refresh Now"):
    st.rerun()

st.sidebar.info("Click button to update live data")
