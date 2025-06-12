import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stock_trading_env1 import StockTradingEnv

# ==== Define Dynamics Model ====
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state, action):
        action = action.view(action.shape[0], -1)  # Giờ action có shape [1000, 2]
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
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
df['feature_volume'] = df['volume'] / df['volume'].rolling(7).max()
df.dropna(inplace=True)

# ==== Create environment ====
def create_env():
    return StockTradingEnv(df=df, render_mode="human")

env = DummyVecEnv([create_env])
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n

# ==== Define model ====
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_tensorboard/")
dynamics_model = DynamicsModel(input_dim=obs_dim, action_dim=2, output_dim=obs_dim + 1)  # 1 for reward
optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)

# ==== Train with Planning ====
real_buffer = []
obs = env.reset()

for timestep in range(1000000):
    action, _ = model.predict(obs)
    next_obs, reward, done, info = env.step(action)

    # Giải nén list vì dùng DummyVecEnv
    obs_np = obs[0]
    next_obs_np = next_obs[0]
    reward_val = reward[0]

    # Lưu vào real buffer
    real_buffer.append((
        obs_np.copy(), 
        np.array(action).copy(), 
        reward_val, 
        next_obs_np.copy()
    ))

    # Update PPO with real data every n steps
    if timestep % 2048 == 0 and timestep > 0:
        model.learn(total_timesteps=2048, reset_num_timesteps=False)

        # ==== Update dynamics model ====
        batch = real_buffer[-1000:]  # recent experiences
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        pred = dynamics_model(states, actions)
        target = torch.cat([next_states, rewards], dim=1)
        loss = nn.MSELoss()(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ==== Planning: Simulate rollouts ====
        simulated_transitions = []
        for _ in range(50):  # K = 50 simulated rollouts
            s_sim = states[np.random.randint(0, len(states))].unsqueeze(0)
            for step in range(5):  # horizon = 5 steps
                a_sim, _ = model.predict(s_sim.detach().numpy())
                a_sim_t = torch.tensor(a_sim, dtype=torch.float32).unsqueeze(0)
                pred_sim = dynamics_model(s_sim, a_sim_t)
                s_next_sim, r_sim = pred_sim[:, :-1], pred_sim[:, -1]
                simulated_transitions.append((
                    s_sim.squeeze(0).detach().numpy(), 
                    a_sim.squeeze(), 
                    r_sim.item(), 
                    s_next_sim.squeeze(0).detach().numpy()
                ))
                s_sim = s_next_sim

        # NOTE: SB3 không hỗ trợ store_transition trực tiếp
        # Bạn cần custom buffer hoặc bỏ qua phần này
        # Nếu có custom replay buffer:
        # for s, a, r, s_next in simulated_transitions:
        #     replay_buffer.add(s, a, r, s_next)

        model.learn(total_timesteps=10000, reset_num_timesteps=False)

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
    if done:
        break
