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
                 take_profit_position_range=(0.10, 0.80),
                 stop_loss_position_range=(0.00, 0.15),
                 max_stop_loss_position=0.30,
                 render_mode=None):

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
        self.take_profit_position_range = take_profit_position_range
        self.stop_loss_position_range = stop_loss_position_range
        self.max_stop_loss_position = max_stop_loss_position
        self.render_mode = render_mode
        self.negative_cash_penalty = 500

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
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
        self.history = []  # tracking hiệu suất và hành động
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

        action_type = action  # 0 = Hold, 1 = Buy, 2 = Sell

        # --- Forced actions ---
        if self.cash < 0 and self.stock_owned > 0:
            action_type = 2
            info['forced_action'] = 'forced_sell_due_to_negative_cash'

        elif self.current_step - self.last_transaction_step >= self.penalty_timeout:
            action_type = 2
            info['forced_action'] = 'forced_sell_due_to_inactivity'

        elif self.stock_owned > 0 and self.avg_buy_price > 0:
            price_change_ratio = (price - self.avg_buy_price) / self.avg_buy_price
            sl_low, sl_high = self.stop_loss_position_range
            if (-price_change_ratio >= sl_low and -price_change_ratio <= sl_high):
                action_type = 2
                info['forced_action'] = 'stop_loss_triggered'

            elif -price_change_ratio > self.max_stop_loss_position:
                action_type = 2
                info['forced_action'] = 'max_stop_loss_triggered'
        # --- Action Execution ---
        if action_type == 1:  # BUY
            amount_to_spend = self.buy_min * self.cash
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

        elif action_type == 2:  # SELL
            shares_to_sell = int(self.sell_min * self.stock_owned)

            if shares_to_sell > 0:
                revenue = shares_to_sell * price * (1 - self.transaction_fee)
                self.cash += revenue
                self.stock_owned -= shares_to_sell
                transaction_occurred = True

                if self.total_buy_shares > 0:
                    price_diff = price - self.avg_buy_price
                    trade_reward = price_diff * shares_to_sell
                    reward += trade_reward * 0.1

        # Cập nhật bước giao dịch trước khi xét phạt
        if transaction_occurred:
            self.last_transaction_step = self.current_step

        inactivity_gap = self.current_step - self.last_transaction_step
        if inactivity_gap >= self.penalty_timeout:
            self.cash -= self.penalty_amount
            reward -= self.penalty_amount
            info['penalty'] = f'inactivity_penalty ({inactivity_gap} steps)'

        # Portfolio evaluation
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

        # --- Logging action ---
        self.history.append({
            'step': self.current_step,
            'action': ['Hold', 'Buy', 'Sell'][action_type],
            'price': price,
            'cash': self.cash,
            'stock': self.stock_owned,
            'portfolio_value': portfolio_value,
            'reward': reward,
            'info': info
        })

        self.current_step += 1
        if self.current_step >= self.total_steps:
            truncated = True

        return self._get_observation(), reward, done, truncated, info

    def render(self):
        if not hasattr(self, 'history') or len(self.history) == 0:
            print("[Render] No history yet.")
            return

        last = self.history[-1]
        print(f"Step: {last['step']}, Action: {last['action']}, Price: {last['price']:.2f}, Cash: {last['cash']:.2f}, Stock: {last['stock']}, Portfolio: {last['portfolio_value']:.2f}, Reward: {last['reward']:.2f}")
        if 'forced_action' in last['info']:
            print("  🚨 Forced action:", last['info']['forced_action'])
        if 'penalty' in last['info']:
            print("  ⚠️ Penalty:", last['info']['penalty'])
