#main_tuned.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
import os

# ایمپورت‌های اختصاصی
from tse_env import TSEPortfolioEnv
from data_loader import fetch_and_clean_data

# ==========================================
# ⚙️ TUNED HYPERPARAMETERS (From Optuna)
# ==========================================
# مقادیر دقیق استخراج شده از خروجی شما
TUNED_PARAMS = {
    'learning_rate': 0.00010010429449913118,
    'gamma': 0.9850740145072168,
    'gae_lambda': 0.9409478201585882,
    'ent_coef': 1.819821976503584e-05,
    'batch_size': 64,
    'n_steps': 1024,  # باید ضریبی از batch_size باشد
}

# معماری شبکه (Large)
POLICY_KWARGS = dict(
    activation_fn=nn.Tanh,
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Large Architecture
)

# تنظیمات کلی
TARGET_TICKERS = ["فولاد", "شپنا", "وبملت", "فملی", "شستا", "خودرو", "وتجارت"]
TOTAL_TIMESTEPS = 200000  # آموزش طولانی‌تر برای همگرایی کامل
MODEL_NAME = "TSE_Genius_Agent_v6_TUNED"

def train_tuned_agent():
    print(f"🚀 Starting Final Training for {MODEL_NAME}...")
    
    # 1. بارگذاری داده‌ها
    print("⏳ Loading Data...")
    data, dates, tickers = fetch_and_clean_data(TARGET_TICKERS)
    
    # تقسیم داده‌ها (برای آموزش نهایی می‌توانیم از کل داده‌های تا قبل از 2023 استفاده کنیم
    # یا حتی کل داده‌ها اگر بخواهیم مدل را برای فردا استفاده کنیم.
    # اما طبق روال علمی، تا 2023 آموزش می‌دهیم تا با تست قبلی قابل مقایسه باشد)
    SPLIT_DATE = "2023-01-01"
    split_idx = -1
    for i, date_str in enumerate(dates):
        if date_str >= SPLIT_DATE:
            split_idx = i
            break
            
    if split_idx == -1: split_idx = int(len(dates) * 0.8)

    # داده‌های آموزش (تا 2023)
    train_data = data[:split_idx]
    train_dates = dates[:split_idx]
    
    print(f"📅 Training Data: {len(train_dates)} days (End: {train_dates[-1]})")

    # 2. ساخت محیط
    env = DummyVecEnv([lambda: TSEPortfolioEnv(
        data=train_data,
        dates=train_dates,
        tickers=tickers,
        initial_amount=1e8,
        transaction_cost_pct=0.0015,
        window_size=30
    )])

    # 3. ساخت مدل با پارامترهای تیون شده
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TUNED_PARAMS['learning_rate'],
        n_steps=TUNED_PARAMS['n_steps'],
        batch_size=TUNED_PARAMS['batch_size'],
        gamma=TUNED_PARAMS['gamma'],
        gae_lambda=TUNED_PARAMS['gae_lambda'],
        ent_coef=TUNED_PARAMS['ent_coef'],
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        tensorboard_log="./ppo_tse_tuned_tensorboard/"
    )

    # 4. شروع آموزش
    print(f"🏋️‍♂️ Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. ذخیره مدل
    model.save(MODEL_NAME)
    print(f"✅ Model saved as '{MODEL_NAME}.zip'")

if __name__ == "__main__":
    train_tuned_agent()
