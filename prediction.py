import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Fetch stock data from Yahoo Finance with fallback for incomplete data


def fetch_stock_data(ticker, years_of_data=2):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() -
                  timedelta(days=years_of_data * 365)).strftime('%Y-%m-%d')

    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        # If no data is returned, raise an error
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {ticker}.")

        # Add Date column and focus on the 'Close' prices
        stock_data['Date'] = stock_data.index
        stock_data = stock_data[['Date', 'Close']]  # Focus on closing prices
        return stock_data

    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return None


# Predict the next 7 days using ARIMA
def predict_with_arima(stock_data, forecast_days):
    stock_data = stock_data.set_index('Date')
    # ARIMA(5,1,0) parameters as an example
    model = ARIMA(stock_data['Close'], order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=forecast_days)
    forecast_dates = pd.date_range(
        stock_data.index[-1], periods=forecast_days+1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    return forecast_df, fit


# # Function to calculate accuracy (Mean Absolute Error)
# def calculate_accuracy(data, predictions):
#     print(type(data), type(predictions))
#     actual = data[-len(predictions):]
#     mae = mean_absolute_error(actual, predictions)
#     return mae

# Function to calculate accuracy (Mean Absolute Error)
def calculate_accuracy(data, predictions):
    # Extract the 'Close' column from the actual data to ensure consistency
    actual = data['Close'][-len(predictions):]

    # Convert predictions to a DataFrame to ensure the same type
    predictions_df = pd.DataFrame({'Forecast': predictions})

    # Now calculate MAE using the same type for both actual and predicted data
    mae = mean_absolute_error(actual, predictions_df['Forecast'])
    return mae


# Streamlit app
def main():
    st.title("Stock Price Prediction for the Next 7 Days")

    # Input: Stock ticker
    ticker = st.text_input("Enter Stock Ticker:")

    if ticker:
        # Fetch stock data
        st.write(f"Fetching data for {ticker}...")
        stock_data = fetch_stock_data(ticker)

        if stock_data is None:
            return  # If data fetch fails, exit

        # Check for missing values
        if stock_data.isnull().any().any():  # This checks for missing values across columns
            stock_data = stock_data.dropna()  # Drop missing values
            st.warning("Missing values were found and removed.")

        # Show data
        st.subheader(f"Last 30 days closing prices for {ticker}")
        st.write(stock_data.tail(30))  # Show last 30 days

        # Perform prediction
        forecast_df, model_fit = predict_with_arima(stock_data, 7)

        # Display the forecasted values for the next 7 days
        st.subheader(f"Predicted closing prices for the next 7 days")
        st.write(forecast_df)

        # Plot the actual data and forecast
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot actual data (stock closing prices)
        ax.plot(stock_data['Date'], stock_data['Close'],
                label='Actual Prices', color='blue')

        # Plot forecasted data (predictions)
        ax.plot(forecast_df['Date'], forecast_df['Forecast'],
                label='Forecasted Prices', color='red', linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.legend()
        st.pyplot(fig)

        # Accuracy (using last 30 days of actual data and the predictions)
        if len(stock_data) > 30:
            predictions = model_fit.forecast(steps=30)
            mae = calculate_accuracy(stock_data[-30:], predictions)
            st.subheader(
                f"Prediction Accuracy (MAE for last 30 days): {mae:.2f}")
            # Explanation of MAE
            st.write("""
                **Mean Absolute Error (MAE)** represents the average of the absolute differences between the actual and predicted values. 
                A lower MAE value indicates a more accurate prediction, meaning the predicted stock prices are closer to the actual ones.
                \n
                The value of MAE is in the same units as the target variable (in this case, stock price), so it gives a direct sense of 
                how much the model's predictions deviate, on average, from the actual values. 
                A MAE value of 0 would indicate perfect predictions, while larger values indicate greater discrepancies.
            """)


if __name__ == "__main__":
    main()
