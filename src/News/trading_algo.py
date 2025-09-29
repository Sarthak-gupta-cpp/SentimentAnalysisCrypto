import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out

model = torch.load("src/News/lstm_model.pth")

class ThresholdTradingAlgoOld:
    def __init__(self, threshold_sell=0.005,threshold_buy=0.005, initial_balance=10000, trade_percent=1.0):
        self.threshold = threshold_buy
        self.threshold_sell = threshold_sell
        self.balance = initial_balance   # cash
        self.btc_holdings = 0.0          # how much BTC you own (positive = long, negative = short liability)
        self.position = 0                # +1 long, -1 short, 0 flat
        self.trade_history = []
        self.portfolio_history = []
        self.nbuys = 0
        self.nsells = 0
        self.nhold = 0

    def step(self, current_price, predicted_change):
        diff = predicted_change*current_price

        # --- BUY signal ---
        if predicted_change > self.threshold and self.position <= 0:
            # If short, close it first
            if self.position == -1:
                # Buy back BTC to close short
                cost_to_close = abs(self.btc_holdings) * current_price
                self.balance -= cost_to_close
                self.btc_holdings = 0
                self.trade_history.append(("CLOSE SHORT", current_price))

            # Go long with all balance
            btc_to_buy = self.balance / current_price
            self.btc_holdings = btc_to_buy
            self.balance = 0
            self.position = 1
            self.trade_history.append(("BUY", current_price))
            self.nbuys += 1

        # --- SELL signal ---
        elif predicted_change < -self.threshold_sell and self.position >= 0:
            # If long, close it first
            if self.position == 1:
                self.balance += self.btc_holdings * current_price
                self.btc_holdings = 0
                self.trade_history.append(("CLOSE LONG", current_price))

            # Go short: borrow BTC & sell immediately
            btc_to_short = self.balance / current_price
            self.btc_holdings = -btc_to_short
            self.balance += btc_to_short * current_price
            self.position = -1
            self.trade_history.append(("SELL SHORT", current_price))
            self.nsells += 1

        else:
            self.trade_history.append(("HOLD", current_price))
            self.nhold += 1
            

    def get_portfolio_value(self, current_price):
        # Cash + (long BTC value or negative liability for short BTC)
        return self.balance + self.btc_holdings * current_price
        
algo = ThresholdTradingAlgoOld(threshold_buy=0.005,threshold_sell=0.005, initial_balance=10000)
algo.portfolio_history = []
for today_price, pred_price in zip(full_test_prices, full_test_pred):
    algo.step(today_price, pred_price)
    pvalue = algo.get_portfolio_value(today_price)
    print("Portfolio Value:", pvalue, " History: ", algo.trade_history[-1], " Prediction: ", pred_price, "actual: ", today_price)
    algo.portfolio_history.append(pvalue)

print("Buys:", algo.nbuys, "Sells:", algo.nsells, "Holds:", algo.nhold)
#plotting graph
plt.figure(figsize=(12,6))
plt.plot(algo.portfolio_history, label='Train Actual Price')
plt.title('Train: Actual vs Predicted Bitcoin Price')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()