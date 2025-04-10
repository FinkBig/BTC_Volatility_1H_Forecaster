# BTC Volatility Predictor

## Overview
This script (`vol_predictoor.py`) forecasts the next-hour volatility and price range for Bitcoin (BTC) using historical data from Binance. It uses an XGBoost model to predict volatility based on features like lagged volatility, EMAs, ATR, funding rates, and open interest. The script generates a terminal report and a plot visualizing the actual and predicted volatility, along with the forecasted price range.

## Features
- Fetches 7 days of hourly OHLCV data, funding rates, and open interest from Binance.
- Trains an XGBoost model to predict the next-hour volatility.
- Calculates in-sample and out-of-sample MAE (Mean Absolute Error) using walk-forward validation.
- Generates a terminal report with the predicted volatility, price range, and model accuracy.
- Saves a plot (`volatility_plot.png`) showing actual vs. predicted volatility and the forecasted price range.

## Requirements
- Python 3.7+
- Required packages (install via `pip`):
  ```bash
  pip install numpy pandas matplotlib xgboost requests tzlocal
