import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env1 import StockTradingEnv

# ==== Load test data ====
df = pd.read_csv('Pep_historical_data_StockScan_test.csv')
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=True)
df['feature_close'] = df['close'].pct_change()
df['feature_open'] = df['open'] / df['close']
df['feature_high'] = df['high'] / df['close']
df['feature_low'] = df['low'] / df['close']
df['feature_volume'] = df['volume'] / df['volume'].rolling(5).max()
### === 2. MOMENTUM FEATURES === ###
window = 5
df["Rolling_Mean_5"] = df["close"].rolling(window=window).mean()
df["Rolling_Std_5"] = df["close"].rolling(window=window).std()
df["Momentum_5"] = df["close"] - df["close"].shift(window)
df["Volatility_5"] = df["feature_close"].rolling(window=window).std()

### === 3. TECHNICAL INDICATORS === ###
# --- RSI (Relative Strength Index) ---
delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-8)
df["RSI_14"] = 100 - (100 / (1 + rs))

# --- MACD and Signal Line ---
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# --- Bollinger Bands (20-day) ---
rolling_mean_20 = df["close"].rolling(window=20).mean()
rolling_std_20 = df["close"].rolling(window=20).std()
df["Bollinger_Upper"] = rolling_mean_20 + 2 * rolling_std_20
df["Bollinger_Lower"] = rolling_mean_20 - 2 * rolling_std_20

### === 4. VOLUME FEATURES === ###
df["Volume_Change"] = df["volume"].pct_change()
df["Volume_Rolling_Mean_5"] = df["volume"].rolling(window=5).mean()
df["Volume_Spike"] = df["volume"] > df["Volume_Rolling_Mean_5"] + 2 * df["volume"].rolling(window=5).std()
df.dropna(inplace=True)

# ==== Create test environment ====
def create_test_env():
    return StockTradingEnv(df=df, render_mode="human")

test_env = DummyVecEnv([create_test_env])

# ==== Load trained model ====
model = PPO.load("ppo_stock_trading_with_planning", env=test_env)

# ==== Run test episode ====
obs = test_env.reset()
total_reward = 0
transaction_log = []
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
    total_reward += reward[0]
    # Ghi log
    current_step = test_env.envs[0].current_step
    price = test_env.envs[0].df.iloc[current_step]['close']
    cash = test_env.envs[0].cash
    stock_owned = test_env.envs[0].stock_owned

    transaction_log.append({
        'step': current_step,
        'date': test_env.envs[0].df.iloc[current_step]['date'],
        'action_type': float(action[0][0]),
        'raw_amount': float(action[0][1]),
        'price': price,
        'cash': cash,
        'stock_owned': stock_owned,
        'reward': reward[0],
        'portfolio_value': cash + stock_owned * price,
        'done': done[0]
    })

    if done[0]:
        break

print(f"\n✅ Total reward on test set: {total_reward:.2f}")

df_log = pd.DataFrame(transaction_log)
df_log.to_csv('predict_transaction_history_test.csv', index=False)

