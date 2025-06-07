import gym
import numpy as np
from gym import spaces
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame,
                 initial_cash=100_000,
                 buy_min=0.44,
                 buy_max=0.62,
                 sell_min=0.22,
                 sell_max=0.96,
                 transaction_fee=0.0,
                 penalty_timeout=12,
                 penalty_amount=2947,
                 min_total_assets=0,
                 max_cash_loss=-5000,
                 win_threshold=1_000_000,
                 render_mode = None):

        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.buy_min = buy_min
        self.buy_max = buy_max
        self.sell_min = sell_min
        self.sell_max = sell_max
        self.transaction_fee = transaction_fee
        self.penalty_timeout = penalty_timeout
        self.penalty_amount = penalty_amount
        self.min_total_assets = min_total_assets
        self.max_cash_loss = max_cash_loss
        self.win_threshold = win_threshold
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

        # Observation: [cash, stock_owned, current_price, volume, high, low]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

        self.reset()

    def _get_observation(self):
        # Bảo vệ chỉ số không vượt giới hạn
        step = min(self.current_step, len(self.df) - 1)

        row = self.df.iloc[step]
        obs = np.array([
            self.cash,
            self.stock_owned,
            row['close'],
            row['volume'],
            row['high'],
            row['low'],
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.current_step = 0
        self.total_steps = len(self.df)
        self.last_transaction_step = 0
        return self._get_observation(), {}

    def step(self, action):
        done = False
        truncated = False
        info = {}

        # Bảo vệ chỉ số không vượt quá giới hạn
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            truncated = True

        price = self.df.iloc[self.current_step]['close']
        transaction_occurred = False
        reward = 0

        # Track giá mua trung bình
        if not hasattr(self, 'avg_buy_price'):
            self.avg_buy_price = 0
            self.total_buy_shares = 0

        if action == 1:  # Buy
            buy_fraction = self.buy_min
            amount_to_spend = buy_fraction * self.cash
            shares_bought = amount_to_spend // price

            if shares_bought > 0:
                cost = shares_bought * price * (1 + self.transaction_fee)
                self.cash -= cost
                self.stock_owned += shares_bought
                transaction_occurred = True

                # Cập nhật avg_buy_price (trung bình giá vốn)
                self.total_buy_shares += shares_bought
                self.avg_buy_price = (
                    (self.avg_buy_price * (self.total_buy_shares - shares_bought) + shares_bought * price)
                    / self.total_buy_shares
                )

        elif action == 2:  # Sell
            sell_fraction = self.sell_min
            shares_to_sell = int(sell_fraction * self.stock_owned)

            if shares_to_sell > 0:
                revenue = shares_to_sell * price * (1 - self.transaction_fee)
                self.cash += revenue
                self.stock_owned -= shares_to_sell
                transaction_occurred = True

                # Phần thưởng/phạt nếu giá bán khác giá mua
                if self.total_buy_shares > 0:
                    price_diff = price - self.avg_buy_price
                    trade_reward = price_diff * shares_to_sell  # lãi/lỗ theo số cổ phiếu bán
                    reward += trade_reward * 0.1  # có thể scale phần thưởng

        if transaction_occurred:
            self.last_transaction_step = self.current_step

        # Phạt nếu quá lâu không giao dịch
        if (self.current_step - self.last_transaction_step) >= self.penalty_timeout:
            self.cash -= self.penalty_amount
            reward -= self.penalty_amount  # cũng ảnh hưởng đến reward

        # Tính tổng tài sản
        portfolio_value = self.cash + self.stock_owned * price

        # Điều kiện dừng
        if portfolio_value >= self.win_threshold:
            done = True
            reward += 1_000_000
        elif portfolio_value < self.min_total_assets or self.cash <= self.max_cash_loss:
            done = True
            reward -= 1_000_000
        else:
            # Cộng thêm phần reward dựa trên tổng tài sản
            reward += portfolio_value / self.initial_cash - 1

        self.current_step += 1
        if self.current_step >= self.total_steps:
            truncated = True

        return self._get_observation(), reward, done, truncated, info


    def render(self):
        step = min(self.current_step, len(self.df) - 1)
        price = self.df.iloc[step]['close']
        total = self.cash + self.stock_owned * price
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Stock: {self.stock_owned}, Price: {price:.2f}, Total: {total:.2f}")
