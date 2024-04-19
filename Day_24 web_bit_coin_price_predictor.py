import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Bank NIFTY Price Predictor App")

stock = "^NSEBANK"#"BTC-USD"

from datetime import datetime
end = datetime.now()
start = datetime(end.year-10,end.month,end.day)

stock = st.text_input("Enter the stock here",stock)

bit_coin_data = yf.download(stock, start, end)

model = load_model("Latest_bit_coin_model.keras")
st.subheader("Bank NIFTY Data")
st.write(bit_coin_data)

splitting_len = int(len(bit_coin_data)*0.9)
x_test = pd.DataFrame(bit_coin_data.Close[splitting_len:])

st.subheader('Original Close Price')
figsize = (15,6)
fig = plt.figure(figsize=figsize)
plt.plot(bit_coin_data.Close,'b')
st.pyplot(fig)

st.subheader("Test Close Price")
st.write(x_test)

st.subheader('Test Close Price')
figsize = (15,6)
fig = plt.figure(figsize=figsize)
plt.plot(x_test,'b')
st.pyplot(fig)

# preprocess the data
# preprocessing the data into [0 to 1] range
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']].values)

x_data = []
y_data = []
for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = bit_coin_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot()

plt.plot(pd.concat([bit_coin_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

st.subheader("Future Price values")
# st.write(ploting_data)

last_100 = bit_coin_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1,1)).reshape(1,-1,1)
prev_100 = np.copy(last_100).tolist()

def predict_future(no_of_days,prev_100):
    future_predictions = []
    for i in range(int(no_of_days)):
        print(type(prev_100))
        next_day = model.predict(prev_100).tolist()
        prev_100[0].append(next_day[0])
        prev_100 = [prev_100[0][1:]]
        future_predictions.append(scaler.inverse_transform(next_day))
    return future_predictions

no_of_days = int(st.text_input("Enter the No of days to be predicted from current date : ","10"))
future_results = predict_future(no_of_days,prev_100)
future_results = np.array(future_results).reshape(-1,1)
print(future_results)
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker = 'o')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel('days')
plt.ylabel('Close Price')
plt.xticks(range(no_of_days))
plt.yticks(range(min(list(map(int, future_results))), max(list(map(int, future_results))),100))
plt.title('Closing Price of Google')
st.pyplot(fig)