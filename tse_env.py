import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class TSEPortfolioEnv(gym.Env):
    """
    نسخه ۵: Quant Insight (چشم‌های باز مدیر دارایی)
    - ورودی‌ها: شامل RSI و Trend (فاصله از SMA) می‌شود.
    - پاداش: بر اساس Risk-Adjusted Return (شبه Sharpe Ratio).
    - هدف: تشخیص نقاط چرخش بازار با استفاده از اندیکاتورها.
    """
    def __init__(self, data, dates, tickers, initial_amount=1e8, transaction_cost_pct=0.0015, window_size=20, diagnosis_mode=False):
        super(TSEPortfolioEnv, self).__init__()
        
        # Data Shape: (Time, Assets, Features)
        # We assume Features[3] is Close Price based on previous files
        self.raw_prices = np.nan_to_num(data[:, :, 3], nan=0.0)
        self.dates = dates
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.diagnosis_mode = diagnosis_mode
        self.log_file = "agent_diagnosis.csv"
        
        # --- FEATURE ENGINEERING (ساخت مغز تحلیلگر) ---
        # محاسبه RSI و Trend برای تمام سهم‌ها در تمام روزها
        print("📊 Engineering Features (RSI, Trend)...")
        self.features_data = self._engineer_features(self.raw_prices)
        # features_data shape: (Time, Assets, 3) -> [Normalized_Price, RSI_Scaled, Trend_Score]
        
        self.max_step = self._find_valid_end_index()
        
        # Action: Weights for (Assets + Cash)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation: (Window, Assets, 3 Features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.n_assets, 3), 
            dtype=np.float32
        )

        if self.diagnosis_mode:
            with open(self.log_file, 'w') as f:
                f.write("Step,Date,Market_Return,Agent_Return,Cash_Ratio,Total_Value,Reward,RSI_Avg\n")

    def _engineer_features(self, prices):
        """
        محاسبه اندیکاتورها برای اینکه ایجنت دید تکنیکال داشته باشد
        """
        n_days, n_assets = prices.shape
        features = np.zeros((n_days, n_assets, 3)) # 3 channels: Return, RSI, Trend
        
        for i in range(n_assets):
            asset_prices = prices[:, i]
            
            # 1. Log Returns (Normalized Price Movement)
            returns = np.diff(np.log(asset_prices + 1e-8), prepend=np.log(asset_prices[0] + 1e-8))
            features[:, i, 0] = returns
            
            # 2. RSI (Relative Strength Index) - 14 Days
            # تشخیص اشباع خرید/فروش
            deltas = np.diff(asset_prices, prepend=asset_prices[0])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.zeros_like(asset_prices)
            avg_loss = np.zeros_like(asset_prices)
            
            # Simple Moving Average for first window, then Wilders smoothing could be used, 
            # but standard SMA is fine for RL context speed
            period = 14
            for t in range(period, n_days):
                avg_gain[t] = np.mean(gains[t-period:t])
                avg_loss[t] = np.mean(losses[t-period:t])
                
            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
            rsi = 100 - (100 / (1 + rs))
            # Scale RSI to [-1, 1] for Neural Network: (RSI - 50) / 50
            features[:, i, 1] = (rsi - 50.0) / 50.0
            
            # 3. Trend (Distance from SMA 20)
            # تشخیص روند صعودی/نزولی
            sma_period = 20
            sma = np.zeros_like(asset_prices)
            for t in range(sma_period, n_days):
                sma[t] = np.mean(asset_prices[t-sma_period:t])
            
            # (Price - SMA) / SMA -> Percentage distance
            trend = np.divide(asset_prices - sma, sma, out=np.zeros_like(asset_prices), where=sma!=0)
            features[:, i, 2] = np.clip(trend * 10, -1, 1) # Scale and clip
            
        return np.nan_to_num(features, nan=0.0)

    def _find_valid_end_index(self):
        limit = len(self.dates) - 2
        for i in range(len(self.dates)):
            prices = self.raw_prices[i, :]
            if np.sum(prices) < 10.0: 
                limit = i - 1
                break
        return limit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size + 20 # Give space for indicators
        self.portfolio_value = self.initial_amount
        self.market_value = self.initial_amount
        
        # Start with Cash
        self.portfolio_weights = np.zeros(self.n_assets + 1)
        self.portfolio_weights[0] = 1.0 
            
        info = {
            'portfolio_value': self.portfolio_value,
            'date': self.dates[self.current_step]
        }
        return self._get_obs(), info

    def step(self, action):
        if self.current_step >= self.max_step:
            return self._get_obs(), 0.0, True, False, self._create_info()

        # --- محاسبه قیمت و ارزش ---
        current_prices = self.raw_prices[self.current_step, :]
        next_prices = self.raw_prices[self.current_step + 1, :]
        
        valid_assets_mask = (current_prices > 10.0) & (next_prices > 10.0)
        safe_current_prices = np.where(current_prices <= 10.0, 1.0, current_prices)

        price_relatives = np.zeros(self.n_assets)
        price_relatives[valid_assets_mask] = (next_prices[valid_assets_mask] - safe_current_prices[valid_assets_mask]) / safe_current_prices[valid_assets_mask]
        
        full_price_relatives = np.concatenate(([0.0], price_relatives)) # Cash is 0% return

        # --- Action ---
        action = np.clip(action, -20, 20) 
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        # --- Transaction Cost ---
        turnover = np.sum(np.abs(weights - self.portfolio_weights))
        transaction_cost = turnover * self.transaction_cost_pct * self.portfolio_value
        
        # --- Portfolio Update ---
        portfolio_growth = np.sum(weights * (1 + full_price_relatives))
        new_value = (self.portfolio_value * portfolio_growth) - transaction_cost
        
        # --- Benchmark Update ---
        if np.sum(valid_assets_mask) > 0:
            avg_market_return = np.mean(price_relatives[valid_assets_mask])
        else:
            avg_market_return = 0.0
        self.market_value = self.market_value * (1 + avg_market_return)

        if new_value <= 0:
            new_value = 1e-8
            done = True
        else:
            done = False

        step_return = (new_value - self.portfolio_value) / self.portfolio_value
        step_return = np.nan_to_num(step_return, nan=0.0)

        self.portfolio_value = new_value
        self.portfolio_weights = weights
        self.current_step += 1
        
        # =================================================================
        # 🚀 REWARD V5: RISK-AWARE MANAGER (SORTINO/SHARPE STYLE)
        # =================================================================
        
        # 1. Excess Return (سود مازاد بر بازار)
        excess_return = step_return - avg_market_return
        
        # 2. Volatility Penalty (جریمه نوسان منفی)
        # اگر بازار در حال ریزش است (Average Trend < 0)، ماندن در سهام جریمه دارد
        avg_trend = np.mean(self.features_data[self.current_step, :, 2])
        market_is_bearish = avg_trend < -0.2
        held_stocks = np.sum(weights[1:]) # چقدر سهام داریم؟
        
        risk_penalty = 0.0
        if market_is_bearish and held_stocks > 0.2:
            # اگر بازار نزولی است و سهام داریم -> جریمه سنگین ریسک
            risk_penalty = held_stocks * 50.0 * abs(avg_market_return)
        
        # 3. Profit Reward
        profit_score = step_return * 100.0
        
        # فرمول نهایی: سود کن، اما اگر بازار خرابه و سهام داری، تنبیه میشی
        reward = profit_score - risk_penalty
        
        # تشویق نقد شدن در شرایط بد
        if market_is_bearish and weights[0] > 0.8:
             reward += 2.0 # آفرین که نقد شدی

        reward = np.clip(reward, -50.0, 50.0)

        # =================================================================

        info = self._create_info()
        
        if self.diagnosis_mode and self.current_step % 50 == 0:
            rsi_avg = np.mean(self.features_data[self.current_step, :, 1])
            self._log_status(avg_market_return, step_return, weights, reward, rsi_avg)

        terminated = (self.current_step >= self.max_step) or done
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # برگرداندن فیچرهای مهندسی شده: [Returns, RSI, Trend]
        return self.features_data[self.current_step - self.window_size : self.current_step]

    def _create_info(self):
        total_portfolio_return_pct = (self.portfolio_value / self.initial_amount - 1) * 100
        total_market_return_pct = (self.market_value / self.initial_amount - 1) * 100
        cash_weight = self.portfolio_weights[0]
        stock_weights = self.portfolio_weights[1:]
        
        return {
            'date': self.dates[self.current_step - 1],
            'portfolio_value': self.portfolio_value,
            'portfolio_return': total_portfolio_return_pct,
            'market_return': total_market_return_pct,
            'cash_balance': self.portfolio_value * cash_weight,
            'allocations': stock_weights,
            'cash_weight': cash_weight
        }

    def _log_status(self, market_ret, agent_ret, weights, reward, rsi_avg):
        try:
            date_str = str(self.dates[self.current_step])
            log_line = f"{self.current_step},{date_str},{market_ret:.4f},{agent_ret:.4f},{weights[0]:.2f},{self.portfolio_value:.0f},{reward:.4f},{rsi_avg:.2f}\n"
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except:
            pass
