import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

st.title(" QuantFlow Dashboard")

symbol = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
data = yf.download(symbol, period="7d", interval="1h")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

st.plotly_chart(fig)
