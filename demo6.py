import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env3 import StockTradingEnv

# ==== Define Dynamics Model for Discrete Action ====
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state, action):
        action_one_hot = torch.nn.functional.one_hot(action.long(), num_classes=3).float()
        x = torch.cat([state, action_one_hot], dim=1)
        return self.model(x)

# ==== Load data ====
df = pd.read_csv('Pep_historical_data_StockScan_train.csv')
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=True)
df['feature_close'] = df['close'].pct_change()
df['feature_open'] = df['open'] / df['close']
df['feature_high'] = df['high'] / df['close']
df['feature_low'] = df['low'] / df['close']
df['feature_volume'] = df['volume'] / df['volume'].rolling(5).max()

window = 5
df["Rolling_Mean_5"] = df["close"].rolling(window=window).mean()
df["Rolling_Std_5"] = df["close"].rolling(window=window).std()
df["Momentum_5"] = df["close"] - df["close"].shift(window)
df["Volatility_5"] = df["feature_close"].rolling(window=window).std()

delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-8)
df["RSI_14"] = 100 - (100 / (1 + rs))

ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

rolling_mean_20 = df["close"].rolling(window=20).mean()
rolling_std_20 = df["close"].rolling(window=20).std()
df["Bollinger_Upper"] = rolling_mean_20 + 2 * rolling_std_20
df["Bollinger_Lower"] = rolling_mean_20 - 2 * rolling_std_20

df["Volume_Change"] = df["volume"].pct_change()
df["Volume_Rolling_Mean_5"] = df["volume"].rolling(window=5).mean()
df["Volume_Spike"] = df["volume"] > df["Volume_Rolling_Mean_5"] + 2 * df["volume"].rolling(window=5).std()
df.dropna(inplace=True)

# ==== Create environment ====
def create_env():
    return StockTradingEnv(df=df, render_mode="human")

env = DummyVecEnv([create_env])
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # use .n for Discrete action space

# ==== Define PPO model ====
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")

# ==== Dynamics model for planning ====
dynamics_model = DynamicsModel(input_dim=obs_dim, action_dim=action_dim, output_dim=obs_dim + 1)
optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)

# ==== Training with planning ====
real_buffer = []
obs = env.reset()

for timestep in range(2000000):
    action, _ = model.predict(obs)
    next_obs, reward, done, info = env.step(action)

    obs_np = obs[0]
    next_obs_np = next_obs[0]
    reward_val = reward[0]
    action_val = int(action) if isinstance(action, (np.ndarray, list)) else action

    real_buffer.append((
        obs_np.copy(), 
        action_val, 
        reward_val, 
        next_obs_np.copy()
    ))

    if timestep % 2048 == 0 and timestep > 0:
        model.learn(total_timesteps=2048, reset_num_timesteps=False)

        # ==== Train dynamics model ====
        batch = real_buffer[-1000:]
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        pred = dynamics_model(states, actions)
        target = torch.cat([next_states, rewards], dim=1)
        loss = nn.MSELoss()(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==== Planning ====
        simulated_transitions = []
        for _ in range(50):
            s_sim = states[np.random.randint(0, len(states))].unsqueeze(0)
            for step in range(5):
                a_sim, _ = model.predict(s_sim.detach().numpy())
                a_sim_tensor = torch.tensor(a_sim, dtype=torch.int64)
                pred_sim = dynamics_model(s_sim, a_sim_tensor)
                s_next_sim, r_sim = pred_sim[:, :-1], pred_sim[:, -1]
                simulated_transitions.append((
                    s_sim.squeeze(0).detach().numpy(),
                    a_sim.squeeze(),
                    r_sim.item(),
                    s_next_sim.squeeze(0).detach().numpy()
                ))
                s_sim = s_next_sim

        model.learn(total_timesteps=2048, reset_num_timesteps=False)

    obs = next_obs
    if done[0]:
        obs = env.reset()

# ==== Save model ====
model.save("ppo_stock_trading_with_planning")

# ==== Test run ====
obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done[0]:
        break
