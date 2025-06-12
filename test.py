import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env1 import StockTradingEnv

# ==== Load test data ====
test_df = pd.read_csv('Pep_historical_data_StockScan_test.csv')
test_df.columns = test_df.columns.str.lower()
test_df['date'] = pd.to_datetime(test_df['date'])
test_df = test_df.sort_values(by='date', ascending=True)
test_df['feature_close'] = test_df['close'].pct_change()
test_df['feature_open'] = test_df['open'] / test_df['close']
test_df['feature_high'] = test_df['high'] / test_df['close']
test_df['feature_low'] = test_df['low'] / test_df['close']
test_df['feature_volume'] = test_df['volume'] / test_df['volume'].rolling(7).max()
test_df.dropna(inplace=True)

# ==== Create test environment ====
def create_test_env():
    return StockTradingEnv(df=test_df, render_mode="human")

test_env = DummyVecEnv([create_test_env])

# ==== Load trained model ====
model = PPO.load("ppo_stock_trading_with_planning", env=test_env)

# ==== Run test episode ====
obs = test_env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = test_env.step(action)
    total_reward += reward[0]
    test_env.render()
    if done[0]:
        break

print(f"\n✅ Total reward on test set: {total_reward:.2f}")

