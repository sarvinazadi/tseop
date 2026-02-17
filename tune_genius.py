import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
import os

# --- ایمپورت‌های اختصاصی شما ---
from tse_env import TSEPortfolioEnv
from data_loader import fetch_and_clean_data

# لیست نمادها (همان لیست اصلی)
TARGET_TICKERS = ["فولاد", "شپنا", "وبملت", "فملی", "شستا", "خودرو", "وتجارت"]
SPLIT_DATE = "2023-01-01"

def load_and_split_data():
    """
    داده‌ها را لود کرده و به دو بخش آموزش و تست تقسیم می‌کند.
    """
    print("⏳ Loading data from PKL...")
    # فراخوانی تابع دیتالودر خودتان
    data, dates, tickers = fetch_and_clean_data(TARGET_TICKERS, force_update=False)
    
    if data is None:
        raise ValueError("❌ Data load failed! Please check clean_market_data.pkl")

    # پیدا کردن ایندکس برش زمانی (2023-01-01)
    split_idx = -1
    for i, date_str in enumerate(dates):
        if date_str >= SPLIT_DATE:
            split_idx = i
            break
    
    if split_idx == -1:
        print("⚠️ Warning: Split date not found. Using 80% split.")
        split_idx = int(len(dates) * 0.8)

    print(f"✂️ Splitting data at index {split_idx} ({dates[split_idx]})")

    # برش داده‌های Numpy
    # data shape: (Days, Assets, Features)
    train_data = data[:split_idx]
    train_dates = dates[:split_idx]
    
    val_data = data[split_idx:]
    val_dates = dates[split_idx:]

    return (train_data, train_dates), (val_data, val_dates), tickers

# --- بارگذاری اولیه داده‌ها (بیرون از حلقه برای سرعت) ---
(train_d, train_dates), (val_d, val_dates), valid_tickers = load_and_split_data()

def objective(trial):
    """
    تابع هدف Optuna: پارامترها را تست می‌کند و بازدهی را برمی‌گرداند.
    """
    
    # 1. پیشنهاد پارامترها (Hyperparameters)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048])
    
    # معماری شبکه (تعداد لایه و نورون)
    net_arch_type = trial.suggest_categorical("net_arch", ["medium", "large"])
    if net_arch_type == "medium":
        net_arch = [dict(pi=[128, 128], vf=[128, 128])]
    else: # large
        net_arch = [dict(pi=[256, 256], vf=[256, 256])]

    policy_kwargs = dict(
        activation_fn=nn.Tanh, # برای مالی Tanh معمولا بهتر است
        net_arch=net_arch
    )

    # 2. ساخت محیط آموزش
    train_env = DummyVecEnv([lambda: TSEPortfolioEnv(
        data=train_d,
        dates=train_dates,
        tickers=valid_tickers,
        initial_amount=1e8,
        transaction_cost_pct=0.0015, # کارمزد واقعی
        window_size=30
    )])

    # 3. ساخت محیط اعتبارسنجی (Validation)
    # مدل را روی داده‌های آینده (بعد از 2023) تست می‌کنیم تا مطمئن شویم واقعا یاد گرفته
    val_env = DummyVecEnv([lambda: TSEPortfolioEnv(
        data=val_d,
        dates=val_dates,
        tickers=valid_tickers,
        initial_amount=1e8,
        transaction_cost_pct=0.0015,
        window_size=30
    )])

    # 4. تعریف مدل
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    # 5. آموزش سریع
    # تعداد استپ کم (مثلا 30,000) فقط برای اینکه ببینیم پتانسیل دارد یا نه
    # در آموزش اصلی (Main) این عدد را 200,000 می‌گذاریم
    try:
        model.learn(total_timesteps=30000)
    except Exception as e:
        print(f"❌ Error in trial: {e}")
        return -1e9 # جریمه سنگین

    # 6. ارزیابی در محیط Validation (آینده)
    # این مهم است: ما پارامتری را می‌خواهیم که در "آینده" سود بدهد
    mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=1)
    
    return mean_reward

if __name__ == "__main__":
    print("\n🚀 Starting Hyperparameter Tuning for TSE Genius V6...")
    print(f"   Target Tickers: {len(valid_tickers)}")
    print(f"   Training Data Days: {len(train_dates)}")
    print(f"   Validation Data Days: {len(val_dates)}")
    
    # ساخت مطالعه (Study)
    study = optuna.create_study(direction="maximize")
    
    # تعداد دورهای تلاش (هر چقدر بیشتر، نتیجه دقیق‌تر اما طولانی‌تر)
    # فعلاً روی ۲۰ می‌گذاریم که حدود ۱۵-۲۰ دقیقه طول بکشد
    study.optimize(objective, n_trials=20)
    
    print("\n✅ Tuning Finished!")
    print("🏆 Best Value (Reward):", study.best_value)
    print("🔧 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # ذخیره در فایل متنی
    with open("best_hyperparameters_v6.txt", "w") as f:
        f.write(str(study.best_params))
    
    print("\n💾 Best params saved to 'best_hyperparameters_v6.txt'")
