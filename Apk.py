import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import streamlit as st


# Starting Date of historical observations
start = '2010-01-01'

# Get the current date and time
current_datetime = datetime.now()
# Extract the current date
end = current_datetime.date()

# Create a sidebar for navigation
st.sidebar.title('Navigation')
selected_page = st.sidebar.radio('Go to', ['Home', 'News and Events', 'About App'])

# Initialize data_training and data_testing as empty DataFrames
data_training = pd.DataFrame()
data_testing = pd.DataFrame()

# Inside the 'News and Events' section
if selected_page == 'News and Events':
    st.title('News and Events')
    st.write('Welcome to the "News and Events" page. Here, you can stay updated with the news and events.')
    # Create a text input for the search term
    search_term = st.text_input("Enter news topic  (e.g., SBI stock , ETH):")
    # Create a button to fetch news
    if not search_term:
        st.warning("Please enter a search term.")
    else:
        try:
            # Construct the URL for Reuters news search
            url = f"https://www.reuters.com/search/news?blob={search_term}"

            # Send an HTTP GET request to the URL
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, "html.parser")

                # Find and display the news headlines and links
                news_elements = soup.find_all("h3", {"class": "search-result-title"})
                if news_elements:
                    st.subheader("Latest News:")
                    for news_element in news_elements:
                        headline = news_element.text.strip()
                        link = news_element.find("a")["href"]
                        full_link = f"https://www.reuters.com{link}"
                        st.write(f"**{headline}**: [{headline}]({full_link})")
                else:
                    st.warning("No news found for the given search term.")
            else:
                st.error(f"Failed to fetch data from Reuters.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


    # You can add more content here, such as recent news articles or upcoming events.


         
    


# Inside the 'About App' section
elif selected_page == 'About App':
    st.title('About App')
    st.write('This is the "About App" page where you can learn more about our stock trend prediction application.')

    st.subheader('How It Works:')
    st.write("""
    - Enter the stock symbol, whether it's a cryptocurrency or a stock like SBIN.NS or BTC-USD.
    - Our app predicts stock trends using historical data and deep learning models.
    - Visualize stock prices and moving averages.
    - Monitor the stock's stability, with an increasing margin indicating stability and a decreasing margin indicating instability.
    - Stay informed with the latest news and market sentiment in the News and Events section.
    - Explore the app's features in the About App section for valuable insights.
             """)

      # Functionality section
    st.subheader('Functionality:')
    st.write("""
    - My app predicts stock trends using long short-term memory (LSTM) models, a class of recurrent neural networks.
    - LSTM models are accurate and excel at identifying trends while filtering out short-term noise.
    - The models in this app have an MSE less than 0.035, ensuring high accuracy.
    - Visualize stock prices with moving averages and spot bullish or bearish trends when these averages intersect.
    - LSTM-based models are a powerful tool for stock trend prediction but should be used alongside other analytical methods and a deep understanding of financial markets.
    - Explore and analyze various stocks effortlessly.
         """)


    st.subheader('Meet the Creater:')
    st.write("Ram Krishna Pandey - Aspiring Data Scientist ")
    st.write("[My email ID](ramkrishnapandey555@gmail.com/)")
    st.write("[My Linkedin profile](https://www.linkedin.com/in/ram-krishna-pandey-8b8620197/)")
    st.write("[My Github profile](https://github.com/ramkrishna1729/)")



# Check if the selected page is 'Home'
if selected_page == 'Home':
    st.title('Trade Trend Tracker')
    
    # Get the user's input for the stock ticker
    user_input = st.text_input('Enter the Stock.Include ".NS" for Indian stock symbols.', 'AAPL')
    
    # Download the stock data using yfinance
    df = yf.download(user_input, start, end)
    
    # Check if the dataframe is not empty
    if not df.empty:
        # Get stock information using yfinance after obtaining user input
        stock_info = yf.Ticker(user_input)
        
        # Retrieve name, sector, and industry information
        sector = stock_info.info.get('sector', 'N/A')
        industry = stock_info.info.get('industry', 'N/A')
        full_name = stock_info.info.get('longName', 'N/A')
        
        # Display the long name, sector, and industry
        st.write(f'Stock Full Name: {full_name}')
        st.write(f'Sector: {sector}')
        st.write(f'Industry: {industry}')
        
        # Display the stock data
        # Describing Data
        st.subheader('Data ranging from 2010 - 2023')
        st.write(df)
    else:
        # Display an error message if the stock ticker is incorrect or data retrieval failed
        st.write(f"Error: No data available for {user_input}. Please check the stock ticker and make sure to include '.NS' for Indian stock symbols.")


   

    # Visualization
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # PLOTTING WITH 100ma
    if df is not None:
        st.subheader('Closing Price vs Time chart with 100 moving average')
        ma100 = df['Close'].rolling(window=100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-Day Moving Average', color='orange')
        plt.plot(df['Close'], label='Closing Price', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)
    else:
        st.write("Error: DataFrame not loaded. Please check your data source.")

    # PLOTTING with 100 and 200 ma
    if df is not None:
        st.subheader('Closing Price vs Time chart with 100 & 200 moving average')
        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='100-Day Moving Average', color='blue')
        plt.plot(ma200, label='200-Day Moving Average', color='red')
        plt.plot(df['Close'], label='Closing Price', color='lightgreen')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)
    else:
        st.write("Error: DataFrame not loaded. Please check your data source.")

    # Splitting Data into Training and Testing again because we want to display only a few days of data
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])  # 70% for training
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])  # 30% for testing

    print(data_training.shape)
    print(data_testing.shape)

    # Scaling the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scaling the data and converting it into an array
    data_training_array = scaler.fit_transform(data_training)

    # Importing ML Model
    import urllib.request
    from keras.models import load_model

    # Specify the raw GitHub URL to the model file
    github_model_url = 'https://github.com/ramkrishna1729/Stock-Trend-Prediction-Web-Apk/raw/master/keras_model.h5'

    try:
        # Download the model file from the GitHub URL
        urllib.request.urlretrieve(github_model_url, 'keras_model.h5')

        # Load the downloaded model
        model = load_model('keras_model.h5')

        # Model is successfully loaded, you can use it here
        print("Model loaded successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

    # Testing Part
    past_100_days = data_testing.head(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)  # this will combine the past 100 days' data and testing data

    x_test = []
    y_test = []

    # Scaling the final_df
    input_data = scaler.fit_transform(final_df)

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])  # represents the first column

    x_test, y_test = np.array(x_test), np.array(y_test)  # Converting it into numpy array

    # Predicting Part
    y_predicted = model.predict(x_test)

    scaler = scaler.scale_
    scaler_factor = 1 / scaler[0]
    y_predicted = y_predicted * scaler_factor
    y_test = y_test * scaler_factor

    # FINAL GRAPH
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label="Actual Price")
    plt.plot(y_predicted, 'r', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
