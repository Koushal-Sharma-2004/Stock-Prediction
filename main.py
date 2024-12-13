import yfinance as yf 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  

#! Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    data['Price Change'] = data['Close'].diff()  #? Calculate daily price change
    data['Direction'] = data['Price Change'].apply(lambda x: 1 if x > 0 else 0)  #? 1 for up, 0 for down
    data.dropna(inplace=True)
    return data

#! Function to fetch recent financial news for a stock
def fetch_financial_news(ticker):
    query = f'{ticker} financial news'
    articles = Article(f'https://news.google.com/search?q={query}')
    articles.download()
    articles.parse()
    return articles.text

#! Perform sentiment analysis using TextBlob
def analyze_sentiment_textblob(news_text):
    analysis = TextBlob(news_text)
    sentiment_score = analysis.sentiment.polarity  #? -1 to 1 scale (negative to positive sentiment)
    return sentiment_score

#! Perform sentiment analysis using VADER
def analyze_sentiment_vader(news_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(news_text)
    return sentiment['compound']  #? Returns a value between -1 and 1

#! Train a RandomForest model
def train_model(data):
    #? Features and target
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Direction']
    
    #? Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    #? Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    #? Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    #? Accuracy check
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, scaler, accuracy

#! Prediction function
def predict_stock_direction(model, scaler, current_data, sentiment_score):
    #? Prepare current day's data for prediction
    features = current_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled[-1].reshape(1, -1))  #? Predict the direction for the latest data point
    
    #? Adjust prediction with sentiment score (e.g., positive sentiment boosts likelihood of an upward trend)
    adjusted_prediction = prediction[0]
    if sentiment_score > 0:
        adjusted_prediction = 1  #? Positive sentiment could lead to an upward movement
    
    return "Stock will go high" if adjusted_prediction == 1 else "Stock will go down"

#! Function to visualize historical stock trends
def visualize_stock_data(data, ticker):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title(f'Historical Stock Trend for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

#! Streamlit Web App
def main():
    st.title("AI Financial Advisor with Stock Prediction")

    #? Input from the user
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ").upper()
    
    if ticker:
        st.write(f"Fetching data for {ticker}...")

        try:
            #? Fetch stock data
            stock_data = fetch_stock_data(ticker)
            
            #? Fetch and analyze financial news
            st.write("Fetching financial news...")
            news = fetch_financial_news(ticker)
            sentiment_score = analyze_sentiment_vader(news)
            st.write(f"Sentiment Score for {ticker}: {sentiment_score}")
            
            #? Visualize stock data
            st.write(f"Visualizing historical stock trends for {ticker}...")
            visualize_stock_data(stock_data, ticker)

            #? Train model
            st.write("Training the model...")
            model, scaler, accuracy = train_model(stock_data)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
            
            #? Make prediction with sentiment consideration
            result = predict_stock_direction(model, scaler, stock_data, sentiment_score)
            st.write(f"Prediction: {result}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()