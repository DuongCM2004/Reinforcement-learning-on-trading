import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from stock_trading_env2 import StockTradingEnv

# ====== DQN và Replay Buffer ======
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=10.0, lr=1e-3,
                 buffer_size=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = 0.9
        self.epsilon_decay = 0.995

        self.action_dim = action_dim

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(state).argmax().item()

    def remember(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s_next, done = self.buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_val = self.model(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target_model(s_next).max(1)[0].unsqueeze(1)
        q_target = r + (1 - done) * self.gamma * q_next

        loss = F.mse_loss(q_val, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ====== Dynamics Model ======
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, action_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state, action):
        action = action.view(action.shape[0], -1)
        x = torch.cat([state, action], dim=1)
        return self.model(x)

# ====== Load và tiền xử lý dữ liệu ======
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

# ====== Tạo môi trường ======
def create_env():
    return StockTradingEnv(df=df, render_mode="human")

env = create_env()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(obs_dim, action_dim)
dynamics_model = DynamicsModel(obs_dim, action_dim, obs_dim + 1)
optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)

# ====== Training ======
real_buffer = []
num_episodes = 200
max_steps = 100000

for episode in range(num_episodes):
    obs, _   = env.reset()
    total_reward = 0

    for t in range(max_steps):
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.remember(obs, action, reward, next_obs, float(done))
        agent.replay()

        real_buffer.append((obs.copy(), action, reward, next_obs.copy()))
        obs = next_obs
        total_reward += reward

        if done:
            agent.update_target_model()
            break

    print(f"Episode {episode} - Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # === Huấn luyện dynamics model ===
    if episode % 10 == 0 and len(real_buffer) >= 1000:
        batch = real_buffer[-1000:]
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        actions_onehot = F.one_hot(actions.squeeze(), num_classes=action_dim).float()
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        pred = dynamics_model(states, actions_onehot)
        target = torch.cat([next_states, rewards], dim=1)
        loss = nn.MSELoss()(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === Mô phỏng rollout ===
        for _ in range(50):
            s_sim = states[np.random.randint(0, len(states))].unsqueeze(0)
            for step in range(5):
                a_sim = agent.act(s_sim.detach().numpy()[0])
                a_sim_oh = F.one_hot(torch.tensor([a_sim]), num_classes=action_dim).float()
                pred_sim = dynamics_model(s_sim, a_sim_oh)
                s_next_sim, r_sim = pred_sim[:, :-1], pred_sim[:, -1]
                agent.remember(s_sim.squeeze(0).detach().numpy(), a_sim, r_sim.item(), s_next_sim.squeeze(0).detach().numpy(), 0.0)
                agent.replay()
                s_sim = s_next_sim

# ====== Save model ======
torch.save(agent.model.state_dict(), "dqn_trading_model.pt")

# ====== Test run ======
obs, _ = env.reset()
for _ in range(1000):
    action = agent.act(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
