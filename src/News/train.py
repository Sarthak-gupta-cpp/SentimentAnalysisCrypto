import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os




# Load sentiment data
sentiment_df = pd.read_csv("lst_price/final_perday_bt.csv")
sentiment_df.rename(columns={"date": "Date"}, inplace=True)
sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

twitter_df = pd.read_csv("lst_price/final_perday_bt_twitter.csv")
twitter_df.rename(columns={"date": "Date"}, inplace=True)
twitter_df["Date"] = pd.to_datetime(twitter_df["Date"])
# sentiment_df = pd.merge(sentiment_df, twitter_df, on="Date", how="inner")

# Load price data
price_df = pd.read_csv("lst_price/BTCUSDT_closing.csv")
price_df.rename(columns={"Open Time": "Date", "Close": "Close_Price"}, inplace=True)
price_df["Date"] = pd.to_datetime(price_df["Date"])



# Merge on Date
merged_df = pd.merge(sentiment_df, price_df, on="Date", how="inner")
merged_df.sort_values("Date", inplace=True)

# Reset index for modeling
merged_df.reset_index(drop=True, inplace=True)

# Add nextday_price_ratio column: next day's price divided by today's price
merged_df['change_pct'] = (merged_df['Close_Price'].shift(-1) -merged_df['Close_Price']) / merged_df['Close_Price']

from sklearn.preprocessing import StandardScaler

SEQ_LEN = 30 # Number of past days to use
features = []
targets = []
dates = []
target2 = []
split_date = np.datetime64('2024-11-10')

for i in range(SEQ_LEN, len(merged_df)):
    past_sentiment = merged_df['compound'].values[i-SEQ_LEN:i]
    past_price = merged_df['change_pct'].values[i-SEQ_LEN:i]
    current_sentiment = merged_df['compound'].values[i]
    current_price = merged_df['Close_Price'].values[i]
    x = np.concatenate([past_sentiment, past_price, [current_sentiment]])
    features.append(x)
    # Target is next day's price change percentage
    if i + 1 < len(merged_df):
        next_price = merged_df['Close_Price'].values[i + 1]
        change_pct = (next_price - current_price) /current_price 
        targets.append(change_pct)
        target2.append(next_price)
        dates.append(merged_df['Date'].values[i + 1])

target3 = [row.Close_Price for _, row in merged_df.iterrows() if row.Date > split_date]

# Remove last feature if it doesn't have a corresponding target
dates = np.array(dates)
if len(features) > len(targets):
    features = features[: len(targets)]
    dates = dates[: len(targets)]

features = np.array(features)
targets = np.array(targets)

# Date-based split
split_date = np.datetime64('2024-11-10')
train_idx = dates < split_date
test_idx = dates >= split_date

X_train = features[train_idx]
X_test = features[test_idx]
y_train = targets[train_idx]
y_test = targets[test_idx]

actual_prices = np.array(target2)
actual_prices2 = np.array(target3)

y_test_prices = actual_prices[test_idx]
# Normalize features and targets
# scaler_X = StandardScaler()
# X_train = scaler_X.fit_transform(X_train)
# X_test = scaler_X.transform(X_test)
# scaler_y = StandardScaler()
# y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


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


input_size = X_train.shape[1]
hidden_size = 32
model = LSTMModel(input_size=input_size, hidden_size=hidden_size)

# --- Training Loop (Train on Train Data) ---
input_size = X_train_tensor.shape[1]
hidden_size = 32
model = LSTMModel(input_size=input_size, hidden_size=hidden_size)

epochs = 3000
lr = 0.00001
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Reshape for LSTM: (batch, seq_len=1, input_size)
X_train_lstm = X_train_tensor.unsqueeze(1)
X_test_lstm = X_test_tensor.unsqueeze(1)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_lstm)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
torch.save(model.state_dict(), "src/News/lstm_model.pth")

