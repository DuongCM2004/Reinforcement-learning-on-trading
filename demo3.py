import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import StockTradingEnv

# Load and preprocess the data
df = pd.read_csv('Pep_historical_data_StockScan.csv')
df.columns = df.columns.str.lower()
print(df.head())
# Create features
df['feature_close'] = df['close'].pct_change()
df['feature_open'] = df['open'] / df['close']
df['feature_high'] = df['high'] / df['close']
df['feature_low'] = df['low'] / df['close']
df['feature_volume'] = df['volume'] / df['volume'].rolling(7*24).max()

df.dropna(inplace=True)
# Create environment factory
def create_env():
    return StockTradingEnv(df=df, render_mode="human")

# Wrap in DummyVecEnv for Stable-Baselines3 compatibility
env = DummyVecEnv([create_env])

# Define model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")

# Train the model
model.learn(total_timesteps= 1000000, tb_log_name="PPO_Trading")

# Save the trained model
model.save("ppo_stock_trading")

# Test run
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break