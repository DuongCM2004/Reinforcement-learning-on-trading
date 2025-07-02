import gym
import numpy as np
from gym import spaces
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame,
                 initial_cash=100_000,
                 buy_fraction=0.5,
                 sell_fraction=0.5,
                 transaction_fee=0.0,
                 penalty_timeout=12,
                 penalty_amount=2947,
                 min_total_assets=0,
                 max_cash_loss=-5000,
                 win_threshold=1_000_000,
                 render_mode=None):

        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.buy_fraction = buy_fraction
        self.sell_fraction = sell_fraction
        self.transaction_fee = transaction_fee
        self.penalty_timeout = penalty_timeout
        self.penalty_amount = penalty_amount
        self.min_total_assets = min_total_assets
        self.max_cash_loss = max_cash_loss
        self.win_threshold = win_threshold
        self.render_mode = render_mode
        self.negative_cash_penalty = 500

        # ✅ Action space: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # ✅ Observation space
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(23,), dtype=np.float32)

        self.reset()

    def _get_observation(self):
        step = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[step]
        obs = np.array([
            self.cash,
            self.stock_owned,
            row['close'],
            row['volume'],
            row['high'],
            row['low'],
            row['feature_close'],
            row['feature_open'],
            row['feature_high'],
            row['feature_low'],
            row['feature_volume'],
            row['Rolling_Mean_5'],
            row['Rolling_Std_5'],
            row['Momentum_5'],
            row['Volatility_5'],
            row['RSI_14'],
            row['MACD'],
            row['MACD_Signal'],
            row['Bollinger_Upper'],
            row['Bollinger_Lower'],
            row['Volume_Change'],
            row['Volume_Rolling_Mean_5'],
            float(row['Volume_Spike'])
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.current_step = 0
        self.total_steps = len(self.df)
        self.last_transaction_step = 0
        self.avg_buy_price = 0
        self.total_buy_shares = 0
        return self._get_observation(), {}

    def step(self, action):
        done = False
        truncated = False
        info = {}

        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            truncated = True

        price = self.df.iloc[self.current_step]['close']
        transaction_occurred = False
        reward = 0

        # ✅ Forced sell if cash is negative
        if self.cash < 0 and self.stock_owned > 0:
            action = 0
            info['forced_action'] = 'forced_sell_due_to_negative_cash'

        if action == 2:  # BUY
            amount_to_spend = self.buy_fraction * self.cash
            shares_bought = int(amount_to_spend // price)

            if shares_bought > 0:
                cost = shares_bought * price * (1 + self.transaction_fee)
                self.cash -= cost
                self.stock_owned += shares_bought
                transaction_occurred = True

                self.total_buy_shares += shares_bought
                self.avg_buy_price = (
                    (self.avg_buy_price * (self.total_buy_shares - shares_bought) + shares_bought * price)
                    / self.total_buy_shares
                )

        elif action == 0:  # SELL
            shares_to_sell = int(self.sell_fraction * self.stock_owned)

            if shares_to_sell > 0:
                revenue = shares_to_sell * price * (1 - self.transaction_fee)
                self.cash += revenue
                self.stock_owned -= shares_to_sell
                transaction_occurred = True

                if self.total_buy_shares > 0:
                    price_diff = price - self.avg_buy_price
                    reward += price_diff * shares_to_sell * 0.1  # reward from gain/loss

        if transaction_occurred:
            self.last_transaction_step = self.current_step

        if (self.current_step - self.last_transaction_step) >= self.penalty_timeout:
            self.cash -= self.penalty_amount
            reward -= self.penalty_amount
            info['penalty'] = 'inactivity_penalty'

        portfolio_value = self.cash + self.stock_owned * price

        if portfolio_value >= self.win_threshold:
            done = True
            reward += 1_000_000
        elif portfolio_value < self.min_total_assets or self.cash <= self.max_cash_loss:
            done = True
            reward -= 1_000_000
        else:
            reward += portfolio_value / self.initial_cash - 1

        if self.cash < 0:
            reward -= self.negative_cash_penalty
            info['penalty'] = 'negative_cash_penalty_applied'

        self.current_step += 1
        if self.current_step >= self.total_steps:
            truncated = True

        return self._get_observation(), reward, done or truncated, info

    def render(self):
        step = min(self.current_step, len(self.df) - 1)
        price = self.df.iloc[step]['close']
        total = self.cash + self.stock_owned * price
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Stock: {self.stock_owned}, Price: {price:.2f}, Total: {total:.2f}")
