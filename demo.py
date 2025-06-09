import pandas as pd

# Load and preprocess the data
df = pd.read_csv('Pep_historical_data_StockScan.csv')
df.columns = df.columns.str.lower()
print(df['date'].head())
# Create features
df['feature_close'] = df['close'].pct_change()
df['feature_open'] = df['open'] / df['close']
df['feature_high'] = df['high'] / df['close']
df['feature_low'] = df['low'] / df['close']
df['feature_volume'] = df['volume'] / df['volume'].rolling(7*24).max()



# # Drop rows with NaNs caused by rolling and pct_change
# df.dropna(inplace=True)

# import gymnasium as gym
# import gym_trading_env  # Ensure this module is installed or available in your path

# # Create the environment
# env = gym.make("TradingEnv",
#     df=df,
#     positions=[-1, 0, 1],  # SHORT, OUT, LONG
#     trading_fees=0.01 / 100,
#     borrow_interest_rate=0.0003 / 100,
# )

# # Run a random agent
# done, truncated = False, False
# observation, info = env.reset()
# while not done and not truncated:
#     action = env.action_space.sample()
#     observation, reward, done, truncated, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")


