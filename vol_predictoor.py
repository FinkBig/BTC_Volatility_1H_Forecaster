import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import requests
import time
from datetime import datetime, timedelta
from tzlocal import get_localzone

# Define the base path for saving the plot (relative to the script directory)
BASE_PATH = "data"
os.makedirs(BASE_PATH, exist_ok=True)  # Create the data directory if it doesn't exist

def fetch_klines(symbol, interval, start_time, end_time, limit=1000):
    """
    Fetch OHLCV data from Binance API.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        interval (str): Time interval (e.g., '1h').
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
        limit (int): Maximum number of data points per request (default: 1000).
    
    Returns:
        pd.DataFrame: OHLCV data with open_time as the index.
    """
    url = 'https://api.binance.com/api/v3/klines'
    data = []
    session = requests.Session()
    while start_time < end_time:
        params = {'symbol': symbol, 'interval': interval, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
        try:
            response = session.get(url, params=params, timeout=10)
            response_data = response.json()
            if not response_data:
                break
            data.extend(response_data)
            start_time = response_data[-1][0] + 1
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OHLCV: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
    session.close()
    if not data:
        raise ValueError("No OHLCV data fetched.")
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_base_volume', 'taker_quote_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    return df[['open', 'high', 'low', 'close']].astype(float)

def fetch_funding_rates(symbol, start_time, end_time, limit=1000):
    """
    Fetch funding rate data from Binance Futures API.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
        limit (int): Maximum number of data points per request (default: 1000).
    
    Returns:
        pd.DataFrame: Funding rate data with fundingTime as the index.
    """
    url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    data = []
    session = requests.Session()
    while True:
        params = {'symbol': symbol, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
        try:
            response = session.get(url, params=params, timeout=10)
            response_data = response.json()
            if not response_data:
                break
            data.extend(response_data)
            if len(response_data) < limit:
                break
            start_time = response_data[-1]['fundingTime'] + 1
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching funding rates: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
    session.close()
    if not data:
        raise ValueError("No funding rate data fetched.")
    df = pd.DataFrame(data)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df.set_index('fundingTime', inplace=True)
    return df[['fundingRate']].astype(float)

def fetch_open_interest_hist(symbol, period, start_time, end_time, limit=500):
    """
    Fetch open interest history from Binance Futures API.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        period (str): Time period (e.g., '1h').
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
        limit (int): Maximum number of data points per request (default: 500).
    
    Returns:
        pd.DataFrame: Open interest data with timestamp as the index.
    """
    url = 'https://fapi.binance.com/futures/data/openInterestHist'
    data = []
    session = requests.Session()
    while start_time < end_time:
        params = {'symbol': symbol, 'period': period, 'startTime': start_time, 'endTime': end_time, 'limit': limit}
        try:
            response = session.get(url, params=params, timeout=10)
            response_data = response.json()
            if not response_data:
                break
            data.extend(response_data)
            if len(response_data) < limit:
                break
            start_time = response_data[-1]['timestamp'] + 1
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching open interest: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
    session.close()
    if not data:
        raise ValueError("No open interest data fetched.")
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df[['sumOpenInterest']].astype(float)

def generate_report(current_time, local_tz, current_price, current_vol, next_hour_pred, lower_bound, upper_bound, mae_in_sample, mae_out_sample):
    """
    Generate a formatted report with the volatility prediction and model accuracy.
    
    Args:
        current_time (pd.Timestamp): Current time in UTC.
        local_tz (str): Local timezone.
        current_price (float): Current BTC price.
        current_vol (float): Realized volatility of the last hour.
        next_hour_pred (float): Predicted volatility for the next hour.
        lower_bound (float): Lower bound of the price range.
        upper_bound (float): Upper bound of the price range.
        mae_in_sample (float): In-sample Mean Absolute Error.
        mae_out_sample (float): Out-of-sample Mean Absolute Error.
    
    Returns:
        str: Formatted report string.
    """
    local_time = current_time.tz_convert(local_tz)
    next_hour_local = (current_time + pd.Timedelta(hours=1)).tz_convert(local_tz)
    tz_name = str(local_tz)
    # Calculate confidence interval for predicted volatility
    vol_lower_bound = max(next_hour_pred - mae_out_sample, 0) * 100  # Convert to percentage
    vol_upper_bound = (next_hour_pred + mae_out_sample) * 100  # Convert to percentage
    return f"""
🤖 BTC Volatility Prediction Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Current Time ({tz_name}): {local_time.strftime('%Y-%m-%d %H:%M:%S')}
💰 BTC Price: ${current_price:,.2f}

📊 Market Conditions:
   • Realized Volatility (Last Hour): {current_vol:.2%}

🔮 Next Hour Prediction (starting at {next_hour_local.strftime('%H:%M:%S')} {tz_name}):
   • Predicted Volatility: {next_hour_pred:.2%}
   • Volatility Range (based on MAE): {vol_lower_bound:.2f}% to {vol_upper_bound:.2f}%
   • Price Range: ${lower_bound:,.2f} to ${upper_bound:,.2f}
   • Range Width: ${(upper_bound - lower_bound):,.2f} ({(upper_bound - lower_bound) / current_price * 100:.2f}% of price)
   • Model Accuracy (MAE In-Sample): {mae_in_sample:.4f}
   • Model Accuracy (MAE Out-of-Sample): {mae_out_sample:.4f}

Generated by VolPredictoor v1.0
"""

def main():
    """
    Main function to fetch data, train the model, and generate the volatility prediction report and plot.
    """
    # Set up time range for the last 7 days
    local_tz = get_localzone()
    current_time = pd.Timestamp.now(tz='UTC')
    start_time = int((current_time - pd.Timedelta(days=7)).timestamp() * 1000)
    end_time = int((current_time + pd.Timedelta(hours=1)).timestamp() * 1000)

    symbol = 'BTCUSDT'
    interval = '1h'
    period = '1h'

    # Fetch data
    spot_df = fetch_klines(symbol, interval, start_time, end_time)
    funding_df = fetch_funding_rates(symbol, start_time, end_time)
    oi_df = fetch_open_interest_hist(symbol, period, start_time, end_time)

    # Prepare DataFrame
    df = spot_df[['open', 'high', 'low', 'close']].copy()
    df['funding_rate'] = funding_df.reindex(df.index, method='ffill')['fundingRate']
    df['open_interest'] = oi_df.reindex(df.index, method='ffill')['sumOpenInterest']

    # Compute volatility and features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = np.abs(df['log_return'])
    # Transform the target variable to reduce the impact of spikes
    df['realized_vol_transformed'] = np.log(df['realized_vol'] + 0.0001)
    df['ema_price_short'] = df['close'].ewm(span=20).mean()
    df['ema_price_long'] = df['close'].ewm(span=60).mean()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    for i in range(1, 6):
        df[f'realized_vol_lag{i}'] = df['realized_vol'].shift(i)
    df = df.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'])

    # Define features and target
    features = ['realized_vol', 'realized_vol_lag1', 'realized_vol_lag2', 'realized_vol_lag3', 'realized_vol_lag4', 'realized_vol_lag5', 'ema_price_short', 'ema_price_long', 'atr_14', 'funding_rate', 'open_interest']
    df = df.dropna()
    X = df[features]
    y = df['realized_vol_transformed'].shift(-1)
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]

    # Train XGBoost model on the full dataset for the final forecast
    model = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.05, reg_lambda=1.0)
    model.fit(X, y)
    predictions_transformed = model.predict(X)
    predictions = np.exp(predictions_transformed) - 0.0001  # Reverse the transformation

    # Calculate in-sample MAE
    actual_vol = np.exp(y) - 0.0001  # Reverse the transformation for actual values
    mae_in_sample = np.mean(np.abs(actual_vol - predictions))

    # Perform walk-forward validation
    training_window = 72  # 3 days of hourly data
    errors = []
    for i in range(len(X) - training_window - 1):
        # Define training and test sets
        X_train = X.iloc[i:i + training_window]
        y_train = y.iloc[i:i + training_window]
        X_test = X.iloc[i + training_window:i + training_window + 1]
        y_test = y.iloc[i + training_window:i + training_window + 1]

        # Train model
        model_wf = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.05, reg_lambda=1.0)
        model_wf.fit(X_train, y_train)

        # Predict and calculate error
        pred_transformed = model_wf.predict(X_test)[0]
        pred = np.exp(pred_transformed) - 0.0001  # Reverse the transformation
        actual_transformed = y_test.iloc[0]
        actual = np.exp(actual_transformed) - 0.0001
        error = np.abs(actual - pred)
        errors.append(error)

    # Calculate out-of-sample MAE
    mae_out_sample = np.mean(errors) if errors else float('nan')

    # Prepare data for plotting
    plot_index = X.index
    next_hour_time = plot_index[-1] + pd.Timedelta(hours=1)
    extended_index = plot_index.append(pd.Index([next_hour_time]))
    next_hour_pred_transformed = model.predict(X.iloc[-1:])[0]
    next_hour_pred = np.exp(next_hour_pred_transformed) - 0.0001  # Reverse the transformation
    extended_predictions = np.append(predictions, next_hour_pred)
    # Use out-of-sample MAE for the volatility range
    upper_bound_vol = next_hour_pred + mae_out_sample
    lower_bound_vol = max(next_hour_pred - mae_out_sample, 0)  # Ensure the lower bound is non-negative

    # Calculate price range for the next hour
    current_price = df['close'].iloc[-1]
    lower_bound_price = current_price - current_price * next_hour_pred
    upper_bound_price = current_price + current_price * next_hour_pred

    # Plot with volatility and price range
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot volatility on the left y-axis
    ax1.plot(plot_index, actual_vol, label='Actual Volatility', color='blue')
    ax1.plot(extended_index, extended_predictions, label='Predicted Volatility', color='orange')
    ax1.fill_between([plot_index[-1], next_hour_time], 
                     [extended_predictions[-2], lower_bound_vol], 
                     [extended_predictions[-2], upper_bound_vol], 
                     color='orange', alpha=0.2, label=f'Volatility Range (±{mae_out_sample:.4f})')
    # Add annotations for volatility range bounds
    ax1.text(next_hour_time, upper_bound_vol, f'{upper_bound_vol*100:.2f}%', color='orange', ha='left', va='bottom')
    ax1.text(next_hour_time, lower_bound_vol, f'{lower_bound_vol*100:.2f}%', color='orange', ha='left', va='top')
    ax1.set_xlabel('Time (UTC)')
    ax1.set_ylabel('Realized Volatility (Absolute Log Return)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot price range on the right y-axis
    ax2 = ax1.twinx()
    ax2.fill_between([plot_index[-1], next_hour_time], 
                     [current_price, lower_bound_price], 
                     [current_price, upper_bound_price], 
                     color='red', alpha=0.3, label='Price Range')
    ax2.set_ylabel('Price (USD)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Set y-axis limits for price to zoom in on the range
    price_range_width = upper_bound_price - lower_bound_price
    ax2.set_ylim(lower_bound_price - price_range_width * 0.5, upper_bound_price + price_range_width * 0.5)
    
    # Add annotations for price range bounds
    ax2.text(next_hour_time, upper_bound_price, f'${upper_bound_price:,.2f}', color='red', ha='left', va='bottom')
    ax2.text(next_hour_time, lower_bound_price, f'${lower_bound_price:,.2f}', color='red', ha='left', va='top')
    
    # Add a vertical line to mark the start of the forecast
    ax1.axvline(x=plot_index[-1], color='gray', linestyle='--', alpha=0.5)
    
    ax2.legend(loc='upper right')
    plt.title('BTC Hourly Realized Volatility and Price Range Forecast')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, 'volatility_plot.png'))
    plt.close()

    # Generate and print report
    current_vol = df['realized_vol'].iloc[-1]
    report = generate_report(current_time, local_tz, current_price, current_vol, next_hour_pred, lower_bound_price, upper_bound_price, mae_in_sample, mae_out_sample)
    print(report)

if __name__ == "__main__":
    main()
