import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files
import tensorflow as tf
import random

# Set random seeds for reproducibility
tf.random.set_seed(20)
np.random.seed(20)
random.seed(20)


# --- 1. Modified Trading Algorithm Class (with percentage threshold) ---
class ThresholdTradingAlgo:
    def __init__(self, threshold_percentage=1.0, initial_balance=10000):
        self.threshold_percentage = threshold_percentage
        self.balance = initial_balance
        self.btc_holdings = 0.0
        self.position = 0
        self.trade_history = []

    def step(self, current_price, predicted_price):
        predicted_change = predicted_price - current_price

        # Calculate the required change in absolute dollars based on the percentage threshold
        required_change = (self.threshold_percentage / 100) * current_price

        # --- BUY signal ---
        if predicted_change > required_change and self.position <= 0:
            if self.position == -1:
                cost_to_close = abs(self.btc_holdings) * current_price
                self.balance -= cost_to_close
                self.btc_holdings = 0

            btc_to_buy = self.balance / current_price
            self.btc_holdings = btc_to_buy
            self.balance = 0
            self.position = 1

        # --- SELL signal ---
        elif predicted_change < -required_change and self.position >= 0:
            if self.position == 1:
                self.balance += self.btc_holdings * current_price
                self.btc_holdings = 0

            btc_to_short = self.balance / current_price
            self.btc_holdings = -btc_to_short
            self.balance += btc_to_short * current_price
            self.position = -1
        else:
            pass


    def get_portfolio_value(self, current_price):
        return self.balance + self.btc_holdings * current_price

# --- 2. Data Loading ---
print("Please upload your 'bitcoin_data 2_data(1).csv.xlsx' file.")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name)
data = df['Close'].values.reshape(-1, 1)

# --- 3. Data Preprocessing ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

n_steps = 60
X, y = create_sequences(scaled_data, n_steps)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 4. Build and Train the LSTM Model (with Dropout and Early Stopping) ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(Dropout(0.2)) # Add a dropout layer to prevent overfitting
model.add(LSTM(units=50))
model.add(Dropout(0.2)) # Add another dropout layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Use Early Stopping to stop training when performance on the validation set stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])

# --- 5. Prediction on Test Data ---
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- 6. Hypertune the Threshold (using percentages) ---
threshold_range = np.arange(0.1, 5.1, 0.1) # Test a range of percentage thresholds
results = {}

for threshold in threshold_range:
    trading_algo = ThresholdTradingAlgo(threshold_percentage=float(threshold), initial_balance=10000)
    portfolio_values = []
    for i in range(len(actuals) - 1):
        current_price = actuals[i][0]
        predicted_price = predictions[i][0]
        trading_algo.step(current_price, predicted_price)
        portfolio_values.append(trading_algo.get_portfolio_value(current_price))

    final_portfolio_value = trading_algo.get_portfolio_value(actuals[-1][0])
    results[threshold] = final_portfolio_value

# --- 7. Analyze and Visualize Results ---
if results:
    best_threshold = max(results, key=results.get)
    best_portfolio_value = results[best_threshold]

    print(f"\nBest Percentage Threshold: {best_threshold:.2f}%")
    print(f"Highest Final Portfolio Value: ${best_portfolio_value:.2f}")

    plt.figure(figsize=(12, 8))
    plt.plot(list(results.keys()), list(results.values()))
    plt.title('Final Portfolio Value vs. Percentage Threshold')
    plt.xlabel('Percentage Threshold (%)')
    plt.ylabel('Final Portfolio Value (USD)')
    plt.grid(True)
    plt.show()

# --- 8. Optional: Run simulation with the best threshold and visualize portfolio value over time ---
if results:
    print(f"\nRunning simulation with the best threshold ({best_threshold:.2f}%) for visualization:")
    trading_algo = ThresholdTradingAlgo(threshold_percentage=float(best_threshold), initial_balance=10000)
    portfolio_values_best_threshold = []
    for i in range(len(actuals) - 1):
        current_price = actuals[i][0]
        predicted_price = predictions[i][0]
        trading_algo.step(current_price, predicted_price)
        portfolio_values_best_threshold.append(trading_algo.get_portfolio_value(current_price))

    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_values_best_threshold, label=f'Portfolio Value (Best Threshold: {best_threshold:.2f}%)')
    plt.title('Portfolio Value Over Time (on Test Data) with Best Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 4. Model Training & Saving (Run once) ---

# Check if the model already exists to avoid retraining
try:
    # Use the new .keras format here
    model = load_model('best_lstm_model.keras')
    print("Loaded previously saved model.")
except:
    # ... rest of model building and training code
    
    # SAVE the model after successful training
    # Use the new .keras format here
    model.save('best_lstm_model.keras')
    print("New model trained and saved successfully.")

