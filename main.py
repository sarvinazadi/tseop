import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import torch.nn as nn

# ایمپورت ماژول‌های خودمان
from tse_env import TSEPortfolioEnv
from data_loader import fetch_and_clean_data

# تنظیمات نمودارها برای نمایش فارسی/زیبا
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('ggplot')

TARGET_TICKERS = ["فولاد", "شپنا", "وبملت", "فملی", "شستا", "خودرو", "وتجارت"]

def run_backtest_and_plot(env, model, valid_tickers):
    """
    این تابع مسئول اجرای Rollout، ذخیره اکسل و رسم نمودار است.
    """
    print("\n📉 Starting Backtest (Rollout)...")
    
    obs, info = env.reset()
    done = False
    
    # لیست‌هایی برای ذخیره تاریخچه
    history = []
    
    while not done:
        # پیش‌بینی اکشن (بدون حالت تصادفی برای تست)
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        # ذخیره اطلاعات این روز
        day_record = {
            'Date': info['date'],
            'Portfolio Value': info['portfolio_value'],
            'Agent Return (%)': info['portfolio_return'],
            'Market Return (%)': info['market_return'],
            'Cash': info['cash_balance'],
            'Reward': reward
        }
        
        # اضافه کردن وزن هر سهم به رکورد (برای اکسل)
        allocations = info['allocations']
        for idx, ticker in enumerate(valid_tickers):
            # اگر تعداد سهم‌ها کمتر از تعداد تیکرها بود (به هر دلیلی)، هندل کن
            if idx < len(allocations):
                day_record[f"Alloc_{ticker}"] = allocations[idx]
        
        history.append(day_record)

        # نمایش وضعیت هر 200 روز
        if len(history) % 200 == 0:
            print(f"   📅 {info['date']} | Value: {info['portfolio_value']:,.0f} | Return: {info['portfolio_return']:.2f}%")

    # تبدیل به دیتافریم
    df_res = pd.DataFrame(history)
    df_res['Date'] = pd.to_datetime(df_res['Date'])
    df_res.set_index('Date', inplace=True)
    
    # -------------------------------------------------------
    # 1. خروجی اکسل
    # -------------------------------------------------------
    excel_path = "backtest_results.xlsx"
    df_res.to_excel(excel_path)
    print(f"\n✅ Excel report saved to: {excel_path}")

    # -------------------------------------------------------
    # 2. رسم نمودار عملکرد (سود ایجنت vs بازار)
    # -------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_res.index, df_res['Agent Return (%)'], label='AI Agent', color='blue', linewidth=2)
    plt.plot(df_res.index, df_res['Market Return (%)'], label='Market (Benchmark)', color='gray', linestyle='--', alpha=0.7)
    
    plt.title("AI Agent vs Market Performance")
    plt.ylabel("Cumulative Return (%)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    
    perf_path = "chart_performance.png"
    plt.savefig(perf_path)
    print(f"📊 Performance chart saved to: {perf_path}")
    plt.show() # نمایش پنجره (اگر روی سیستم لوکال هستید)

    # -------------------------------------------------------
    # 3. رسم نمودار تخصیص دارایی (Asset Allocation Area Chart)
    # -------------------------------------------------------
    # انتخاب ستون‌های مربوط به تخصیص
    alloc_cols = [c for c in df_res.columns if c.startswith("Alloc_")]
    
    if alloc_cols:
        plt.figure(figsize=(12, 6))
        plt.stackplot(df_res.index, df_res[alloc_cols].T, labels=[c.replace("Alloc_", "") for c in alloc_cols], alpha=0.8)
        plt.title("Portfolio Asset Allocation Over Time")
        plt.ylabel("Allocation Ratio (0.0 to 1.0)")
        plt.xlabel("Date")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        alloc_path = "chart_allocation.png"
        plt.savefig(alloc_path)
        print(f"🎨 Allocation chart saved to: {alloc_path}")
        plt.show()

    # چاپ خلاصه نهایی
    final_val = df_res['Portfolio Value'].iloc[-1]
    final_ret = df_res['Agent Return (%)'].iloc[-1]
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Initial Wealth: 100,000,000")
    print(f"   Final Wealth:   {final_val:,.0f}")
    print(f"   Total Return:   {final_ret:.2f}%")


def main():
    # 1. لود دیتا
    print("🚀 Loading data...")
    data, dates, valid_tickers = fetch_and_clean_data(TARGET_TICKERS, force_update=False)
    
    if data is None:
        return

    # 2. محیط
    env = TSEPortfolioEnv(
        data=data,
        dates=dates,
        tickers=valid_tickers,
        initial_amount=1e8,
        transaction_cost_pct=0.0015,
        window_size=30
    )

# تعریف ساختار مغز جدید (کمی بزرگتر و عمیق‌تر)
# net_arch=[dict(pi=[128, 128], vf=[128, 128])] یعنی:
# دو لایه ۱۲۸ تایی برای تصمیم‌گیری (Policy)
# دو لایه ۱۲۸ تایی برای تخمین ارزش (Value Function)
# اکتیویشن Tanh برای داده‌های مالی معمولا بهتر از ReLU عمل می‌کند
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    # 3. آموزش مدل
    TOTAL_TIMESTEPS = 200_000 
    print(f"\n🧠 Training PPO Agent ({TOTAL_TIMESTEPS} steps)...")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,

        learning_rate=2e-4,
        n_steps=2048,
        gamma=0.99,
        batch_size=64,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_tse_logs/"
    )
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        print("✅ Training Finished.")
        model.save("ppo_tse_agent_final")
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return

    # 4. اجرای بک‌تست و خروجی‌ها (جامپ به نمودار و اکسل)
    # محیط را برای تست ریست می‌کنیم (می‌توانید محیط جداگانه‌ای هم بسازید)
    env.diagnosis_mode = False # لاگ‌های متنی را خاموش می‌کنیم تا سرعت زیاد شود
    run_backtest_and_plot(env, model, valid_tickers)

# ==========================================
# بخش ذخیره‌سازی مدل (The Save Protocol)
# ==========================================

# 1. ذخیره خود مدل (مغز ایجنت)
    model_name = "TSE_Genius_Agent_v5"
    model.save(model_name)
    print(f"✅ Model saved successfully as: {model_name}.zip")

    # 2. ذخیره کردن متغیرهای نرمال‌سازی (اگر از VecNormalize استفاده کرده باشید مهم است)
    # اگر از VecNormalize استفاده نکردید، این بخش ارور می‌دهد که مشکلی نیست.
    try:
        env.save("vec_normalize.pkl")
        print("✅ Environment normalization stats saved.")
    except:
        pass

    print("--- فرایند ذخیره‌سازی تکمیل شد ---")

if __name__ == "__main__":
    main()
