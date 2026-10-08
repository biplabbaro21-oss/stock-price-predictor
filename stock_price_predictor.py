# stock_price_predictor.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Download historical stock data (Apple as example)
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Step 2: Prepare data
data.reset_index(inplace=True)
data['Days'] = (data.index)  # simple numeric index for regression

X = np.array(data['Days']).reshape(-1, 1)
y = data['Close']

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict closing prices
data['Predicted'] = model.predict(X)

# Step 5: Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Close'], label="Actual Price", color="blue")
plt.plot(data['Date'], data['Predicted'], label="Predicted Price", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("Stock Price Prediction (Apple)")
plt.legend()
plt.show()
