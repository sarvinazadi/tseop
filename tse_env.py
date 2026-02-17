#tse_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TSEPortfolioEnv(gym.Env):
    """
    نسخه ۶: Genius Mode (V5 Skeleton + V6 Features)
    - ساختار: دقیقاً مشابه نسخه ۵ (V5) که ۷۳۰٪ سود داد.
    - ارتقا: اضافه شدن حجم (Volume) و نوسان (Volatility) به ورودی‌ها.
    - ورودی‌ها (۵ عدد): [LogReturn, RSI, Trend, Vol_Ratio, Volatility]
    - پاداش: همان پاداش Risk-Adjusted موفق نسخه ۵.
    """
    def __init__(self, data, dates, tickers, initial_amount=1e8, transaction_cost_pct=0.0015, window_size=20, diagnosis_mode=False):
        super(TSEPortfolioEnv, self).__init__()
        
        # Data Shape: (Time, Assets, Features)
        # Features: 0:Open, 1:High, 2:Low, 3:Close, 4:Volume (اگر موجود باشد)
        self.raw_prices = np.nan_to_num(data[:, :, 3], nan=0.0)
        
        # --- FEATURE 4: استخراج حجم ---
        # بررسی می‌کنیم آیا ستون حجم (اینکس ۴) در دیتا وجود دارد یا خیر
        if data.shape[2] > 4:
            self.raw_volumes = np.nan_to_num(data[:, :, 4], nan=0.0)
        else:
            print("⚠️ Warning: Volume data not found! Using zeros.")
            self.raw_volumes = np.zeros_like(self.raw_prices)

        self.dates = dates
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.diagnosis_mode = diagnosis_mode
        self.log_file = "agent_diagnosis.csv"
        
        # --- FEATURE ENGINEERING (مغز تحلیلگر V6) ---
        print("📊 Engineering Features V6 (RSI, Trend, Volume, Volatility)...")
        self.features_data = self._engineer_features(self.raw_prices, self.raw_volumes)
        # features_data shape: (Time, Assets, 5)
        
        self.max_step = self._find_valid_end_index()
        
        # Action: Weights for (Assets + Cash)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation: (Window, Assets, 5 Features) -> تغییر سایز به ۵
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.n_assets, 5), 
            dtype=np.float32
        )

        if self.diagnosis_mode:
            with open(self.log_file, 'w') as f:
                f.write("Step,Date,Market_Return,Agent_Return,Cash_Ratio,Total_Value,Reward,RSI_Avg\n")

    def _engineer_features(self, prices, volumes):
        """
        محاسبه ۵ اندیکاتور کلیدی (ترکیب تکنیکال و رفتار بازار)
        """
        n_days, n_assets = prices.shape
        features = np.zeros((n_days, n_assets, 5)) # 5 channels
        
        for i in range(n_assets):
            asset_prices = prices[:, i]
            asset_vols = volumes[:, i]
            
            # --- 1. Log Returns (بازدهی) ---
            returns = np.diff(np.log(asset_prices + 1e-8), prepend=np.log(asset_prices[0] + 1e-8))
            features[:, i, 0] = returns
            
            # --- 2. RSI (قدرت نسبی) ---
            deltas = np.diff(asset_prices, prepend=asset_prices[0])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.zeros_like(asset_prices)
            avg_loss = np.zeros_like(asset_prices)
            period = 14
            
            for t in range(period, n_days):
                avg_gain[t] = np.mean(gains[t-period:t])
                avg_loss[t] = np.mean(losses[t-period:t])
                
            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
            rsi = 100 - (100 / (1 + rs))
            features[:, i, 1] = (rsi - 50.0) / 50.0
            
            # --- 3. Trend (فاصله از میانگین ۲۰ روزه) ---
            sma_period = 20
            sma = np.zeros_like(asset_prices)
            for t in range(sma_period, n_days):
                sma[t] = np.mean(asset_prices[t-sma_period:t])
            
            trend = np.divide(asset_prices - sma, sma, out=np.zeros_like(asset_prices), where=sma!=0)
            features[:, i, 2] = np.clip(trend * 10, -1, 1)
            
            # --- 4. Volume Ratio (جدید در V6) ---
            # نسبت حجم امروز به میانگین حجم ۲۰ روز گذشته (ورود پول هوشمند)
            vol_sma = np.zeros_like(asset_vols)
            for t in range(sma_period, n_days):
                vol_sma[t] = np.mean(asset_vols[t-sma_period:t])
            
            vol_ratio = np.divide(asset_vols, vol_sma + 1e-8, out=np.zeros_like(asset_vols))
            # نرمال سازی: عدد ۱ یعنی نرمال، ۵ یعنی حجم ۵ برابر
            features[:, i, 3] = np.clip(vol_ratio, 0, 5) / 5.0
            
            # --- 5. Volatility (جدید در V6) ---
            # انحراف معیار بازدهی‌های ۲۰ روزه (تشخیص ریسک)
            volatility = np.zeros_like(asset_prices)
            for t in range(sma_period, n_days):
                window_rets = returns[t-sma_period:t]
                volatility[t] = np.std(window_rets)
            
            features[:, i, 4] = np.clip(volatility * 10, 0, 1)

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
        self.current_step = self.window_size + 20 
        self.portfolio_value = self.initial_amount
        self.market_value = self.initial_amount
        
        self.portfolio_weights = np.zeros(self.n_assets + 1)
        self.portfolio_weights[0] = 1.0 
            
        # FIX: Ensure keys exist for initial step to avoid KeyError
        info = {
            'portfolio_value': self.portfolio_value,
            'date': self.dates[self.current_step],
            'portfolio_return': 0.0,
            'market_return': 0.0
        }
        return self._get_obs(), info

    def step(self, action):
        if self.current_step >= self.max_step:
            return self._get_obs(), 0.0, True, False, self._create_info()

        # --- محاسبه قیمت ---
        current_prices = self.raw_prices[self.current_step, :]
        next_prices = self.raw_prices[self.current_step + 1, :]
        
        valid_assets_mask = (current_prices > 10.0) & (next_prices > 10.0)
        safe_current_prices = np.where(current_prices <= 10.0, 1.0, current_prices)

        price_relatives = np.zeros(self.n_assets)
        price_relatives[valid_assets_mask] = (next_prices[valid_assets_mask] - safe_current_prices[valid_assets_mask]) / safe_current_prices[valid_assets_mask]
        
        full_price_relatives = np.concatenate(([0.0], price_relatives)) 

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
        # 🚀 REWARD V5: RISK-AWARE MANAGER (حفظ شده برای V6)
        # =================================================================
        excess_return = step_return - avg_market_return
        
        # محاسبه ترند بازار برای پاداش
        avg_trend = np.mean(self.features_data[self.current_step, :, 2])
        market_is_bearish = avg_trend < -0.2
        held_stocks = np.sum(weights[1:]) 
        
        risk_penalty = 0.0
        if market_is_bearish and held_stocks > 0.2:
            risk_penalty = held_stocks * 50.0 * abs(avg_market_return)
        
        profit_score = step_return * 100.0
        reward = profit_score - risk_penalty
        
        if market_is_bearish and weights[0] > 0.8:
             reward += 2.0 

        reward = np.clip(reward, -50.0, 50.0)
        # =================================================================

        # ساخت Info کامل برای جلوگیری از KeyError در main.py
        info = self._create_info()
        
        if self.diagnosis_mode and self.current_step % 50 == 0:
            rsi_avg = np.mean(self.features_data[self.current_step, :, 1])
            self._log_status(avg_market_return, step_return, weights, reward, rsi_avg)

        terminated = (self.current_step >= self.max_step) or done
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # برگرداندن ۵ فیچر به جای ۳
        return self.features_data[self.current_step - self.window_size : self.current_step]

    def _create_info(self):
        """
        ساخت دیکشنری اطلاعات برای لاگ کردن در main.py
        """
        total_portfolio_return_pct = (self.portfolio_value / self.initial_amount - 1) * 100
        total_market_return_pct = (self.market_value / self.initial_amount - 1) * 100
        cash_weight = self.portfolio_weights[0]
        stock_weights = self.portfolio_weights[1:]
        
        return {
            'date': self.dates[self.current_step - 1],
            'portfolio_value': self.portfolio_value,
            'portfolio_return': total_portfolio_return_pct, # کلید حیاتی برای main.py
            'market_return': total_market_return_pct,
            'cash_balance': self.portfolio_value * cash_weight,
            'allocations': stock_weights,
            'cash_weight': cash_weight
        }

    def _log_status(self, market_ret, agent_ret, weights, reward, rsi_avg):
        try:
            date_str = str(self.dates[self.current_step])
            l_line = f"{self.current_step},{date_str},{market_ret:.4f},{agent_ret:.4f},{weights[0]:.2f},{self.portfolio_value:.0f},{reward:.4f},{rsi_avg:.2f}\n"
            with open(self.log_file, 'a') as f:
                f.write(l_line)
        except:
            pass
