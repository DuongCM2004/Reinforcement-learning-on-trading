import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stock_trading_env2 import StockTradingEnv

# ==== Define your DQN model (same architecture used during training) ====
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ==== Load test data ====
df = pd.read_csv('Pep_historical_data_StockScan_test1.csv')
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=True)
df['feature_close'] = df['close'].pct_change()
df['feature_open'] = df['open'] / df['close']
df['feature_high'] = df['high'] / df['close']
df['feature_low'] = df['low'] / df['close']
df['feature_volume'] = df['volume'] / df['volume'].rolling(5).max()

window = 5
df['Rolling_Mean_5'] = df['close'].rolling(window=window).mean()
df['Rolling_Std_5'] = df['close'].rolling(window=window).std()
df['Momentum_5'] = df['close'] - df['close'].shift(window)
df['Volatility_5'] = df['feature_close'].rolling(window=window).std()

delta = df['close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-8)
df['RSI_14'] = 100 - (100 / (1 + rs))

df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

rolling_mean_20 = df['close'].rolling(window=20).mean()
rolling_std_20 = df['close'].rolling(window=20).std()
df['Bollinger_Upper'] = rolling_mean_20 + 2 * rolling_std_20
df['Bollinger_Lower'] = rolling_mean_20 - 2 * rolling_std_20

df['Volume_Change'] = df['volume'].pct_change()
df['Volume_Rolling_Mean_5'] = df['volume'].rolling(window=5).mean()
df['Volume_Spike'] = df['volume'] > df['Volume_Rolling_Mean_5'] + 2 * df['volume'].rolling(window=5).std()
df.dropna(inplace=True)

# ==== Create test environment ====
env = StockTradingEnv(df=df, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # ✅ vì là Discrete, dùng .n thay vì .shape[0]

# ==== Load DQN model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dqn_trading_model.pt", map_location=device))
model.eval()

# ==== Run test episode ====
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    obs, reward, done, info = env.step(action)
    total_reward += reward
    state = obs
    env.render()

print(f"\n✅ Total reward on test set: {total_reward:.2f}")
