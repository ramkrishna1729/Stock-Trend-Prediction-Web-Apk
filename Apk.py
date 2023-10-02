import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

#Taking a bigger data set improves the accuracy of the model 
#start = '2010-01-01'
#end = '2019-12-31'

start = '2010-01-01'
end = '2023-09-29'


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter the Stock Ticker','AAPL')
import yfinance as yf
df = yf.download(user_input, start ,end)
df.head()  

#Decribing Data
st.subheader('Data from 2021 -2023')
st.write(df.describe())

#Visualization 
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

#PLOTTING WITH 100ma
# Check if the DataFrame is loaded successfully
if df is not None:
    st.subheader('Closing Price vs Time chart with 100 moving average')
    
    # Calculate the 100-day moving average
    ma100 = df['Close'].rolling(window=100).mean()  # Use parentheses to invoke the mean method

    # Create a figure and plot the data
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Display the plot using Streamlit's pyplot functionality
    st.pyplot(fig)
else:
    st.write("Error: DataFrame not loaded. Please check your data source.")


#PLOTING with 100 and 200 ma
# Check if the DataFrame is loaded successfully
if df is not None:
    st.subheader('Closing Price vs Time chart with 100 & 200 moving average')
    
    # Calculate the 100-day moving average
    ma100 = df['Close'].rolling(window=100).mean()  # Use parentheses to invoke the mean method
    ma200 = df['Close'].rolling(window=200).mean()  

    # Create a figure and plot the data
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100-Day Moving Average', color='blue')
    plt.plot(ma200, label='200-Day Moving Average', color='red')
    plt.plot(df['Close'], label='Closing Price', color='green')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Display the plot using Streamlit's pyplot functionality
    st.pyplot(fig)
else:
    st.write("Error: DataFrame not loaded. Please check your data source.")



#Spliting Data into Training and Testing again because we want to display only few days of data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) # using 0-70 % of total values from 'Close' for training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) # using 70-% of total values from 'Close' for testing

print(data_training.shape)
print(data_testing.shape)

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler( feature_range = (0,1))

#scaling the data and converitng it into array
data_training_array = scaler.fit_transform(data_training) 


#Splitting Data in x_train and y_train
#x_train = []
#y_train = []

#for i in range(100,data_training_array.shape[0]) : #making it dynamic
   # x_train.append(data_training_array[i-100:i]) #starting from 0 to i
   # y_train.append(data_training_array[i,0])  # it will have only one column


#LOADING MY model
model = load_model('F:\\Program Data\\Neural network\\keras_model.h5')

#Testing Part
past_100_days = data_testing.head(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True) #this will combine the past 100days data and testing data

x_test = []
y_test = []

#Scaling the final_df
input_data = scaler.fit_transform(final_df)

for i in range(100 , input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])  # represents first column

x_test, y_test = np.array(x_test), np.array(y_test) #Converting it into numpy array

#Predicting Part
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scaler_factor = 1 /scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test *scaler_factor


#FINAL GRAPH
st.subheader('Predictions vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label = "Actual Price")
plt.plot(y_predicted, 'r',label = "predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
