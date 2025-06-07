import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env import StockTradingEnv

# Load and preprocess data
df = pd.read_csv("Pep_historical_data_StockScan.csv")
df.columns = [c.lower() for c in df.columns]  # Ensure lowercase columns
df = df.sort_values("date").reset_index(drop=True)

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