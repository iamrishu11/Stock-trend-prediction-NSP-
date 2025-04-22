import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit page layout
st.set_page_config(layout="wide")
st.title('📈 Stock Trend Predictor')

# Load model from local directory
model_path = 'Stock Predictions Model.keras'
model = load_model(model_path)

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG, MSFT)', 'GOOG')
start = '2013-01-01'
end = '2024-12-31'

if stock:
    # Fetch data
    data = yf.download(stock, start, end)

    st.subheader('Raw Stock Data')
    st.dataframe(data.tail())

    # Split data
    data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    past_100_days = data_train.tail(100)
    final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
    input_data = scaler.fit_transform(final_test_data)

    # MA50
    st.subheader('Price vs MA50')
    ma_50 = data['Close'].rolling(50).mean()
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(ma_50, 'r', label='MA50')
    plt.plot(data['Close'], 'g', label='Closing Price')
    plt.legend()
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.pyplot(fig1)

    # MA50 vs MA100
    st.subheader('Price vs MA50 vs MA100')
    ma_100 = data['Close'].rolling(100).mean()
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(ma_50, 'r', label='MA50')
    plt.plot(ma_100, 'b', label='MA100')
    plt.plot(data['Close'], 'g', label='Closing Price')
    plt.legend()
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.pyplot(fig2)

    # MA100 vs MA200
    st.subheader('Price vs MA100 vs MA200')
    ma_200 = data['Close'].rolling(200).mean()
    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(ma_100, 'r', label='MA100')
    plt.plot(ma_200, 'b', label='MA200')
    plt.plot(data['Close'], 'g', label='Closing Price')
    plt.legend()
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.pyplot(fig3)

    # Prediction input
    x, y = [], []
    for i in range(100, input_data.shape[0]):
        x.append(input_data[i-100:i])
        y.append(input_data[i, 0])
    x, y = np.array(x), np.array(y)

    # Prediction
    predictions = model.predict(x)
    scale_factor = 1 / scaler.scale_[0]
    predictions = predictions * scale_factor
    y = y * scale_factor

    # Prediction graph
    st.subheader('Original vs Predicted Prices')
    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(y, 'g', label='Original Price')
    plt.plot(predictions, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.pyplot(fig4)
