import pandas as pd
import pytse_client as tse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tse_env import TSEPortfolioEnv
import numpy as np

# ==========================================
# تنظیمات
# ==========================================
MODEL_PATH = "TSE_Genius_Agent_v5.zip"
START_DATE_TEST = "2023-01-01"

# لیست نمادها (دقیقاً 7 نماد که در آموزش بوده اند)
# "وتجارت" حذف شد چون در آموزش نبوده است
TICKERS = ["فولاد", "خودرو", "شپنا", "شستا", "وبملت", "فارس", "رمپنا"]

# نگاشت نام‌ها
TICKER_MAP = {
    "فولاد": "FOLD", "خودرو": "KHOD", "شپنا": "SHIP",
    "شستا": "SHTA", "وبملت": "VMLT", "فارس": "FARS", "رمپنا": "RMPN"
}

ENG_TICKERS = [TICKER_MAP[t] for t in TICKERS]

# ==========================================
# 1. تابع ساخت دیتای 3 بعدی (TENSOR)
# ==========================================
def prepare_tensor_data():
    print("📥 Downloading raw data...")
    dfs = []
    
    # 1. دانلود و ادغام دیتافریم‌ها
    for symbol in TICKERS:
        eng_name = TICKER_MAP[symbol]
        try:
            # دانلود دیتا
            ticker_data = tse.download(symbols=symbol, adjust=True)
            df = ticker_data[symbol]
            
            # نرمال‌سازی نام ستون‌ها
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['date', f'{eng_name}_open', f'{eng_name}_high', f'{eng_name}_low', f'{eng_name}_close', f'{eng_name}_volume']
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # حذف داده‌های پرت احتمالی (قیمت صفر)
            df = df[df[f'{eng_name}_close'] > 10]
            
            dfs.append(df)
            print(f"   ✅ {symbol} loaded.")
        except Exception as e:
            print(f"   ❌ Error loading {symbol}: {e}")
            # اگر سهمی لود نشد، کل پروسه فیل می‌شود چون شکل تنسور به هم می‌ریزد
            exit()

    print("🔄 Merging and aligning data...")
    merged_df = pd.concat(dfs, axis=1)
    merged_df = merged_df.ffill().fillna(0) # پر کردن روزهای تعطیل
    merged_df = merged_df.reset_index()
    merged_df = merged_df.rename(columns={'index': 'date'})
    merged_df = merged_df.sort_values('date')

    # فیلتر کردن برای بازه تست
    test_df = merged_df[merged_df['date'] >= START_DATE_TEST].copy()
    
    if len(test_df) == 0:
        print("❌ Error: No data found after start date.")
        exit()

    dates = test_df['date'].dt.strftime('%Y-%m-%d').tolist()
    print(f"📅 Test Range: {dates[0]} -> {dates[-1]} ({len(dates)} days)")

    # 2. تبدیل به آرایه 3 بعدی (Time, Assets, Features)
    n_timesteps = len(test_df)
    n_assets = len(ENG_TICKERS) # باید 7 باشد
    n_features = 5 
    
    tensor_data = np.zeros((n_timesteps, n_assets, n_features))
    
    print(f"🏗 Constructing 3D Numpy Tensor with shape: ({n_timesteps}, {n_assets}, {n_features})...")
    for i, eng_ticker in enumerate(ENG_TICKERS):
        cols = [
            f'{eng_ticker}_open',
            f'{eng_ticker}_high',
            f'{eng_ticker}_low',
            f'{eng_ticker}_close', # Index 3 (Important for Environment)
            f'{eng_ticker}_volume'
        ]
        tensor_data[:, i, :] = test_df[cols].values

    return tensor_data, dates

# ==========================================
# 2. اجرای تست
# ==========================================

if __name__ == "__main__":
    # الف) آماده‌سازی دیتا
    data_tensor, date_list = prepare_tensor_data()

    # ب) ساخت محیط
    print("\n🛠 Initializing Environment (V5 - Quant Insight)...")
    try:
        env = TSEPortfolioEnv(
            data=data_tensor,   
            dates=date_list,    
            tickers=ENG_TICKERS, 
            window_size=30
        )
    except Exception as e:
        print(f"❌ Environment Init Error: {e}")
        exit()

    # ج) لود مدل
    print(f"🧠 Loading Agent: {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: {MODEL_PATH} not found.")
        exit()

    # بررسی تطابق محیط و مدل
    print(f"ℹ️  Model expects observation shape: {model.observation_space.shape}")
    print(f"ℹ️  Environment provides observation shape: {env.observation_space.shape}")

    # د) حلقه اصلی اجرا
    print("\n🎬 Running Simulation on Unseen Data...")
    obs, _ = env.reset()
    done = False
    
    portfolio_values = []
    cash_ratios = []

    while not done:
        # پیش‌بینی اکشن
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        cash_ratios.append(info['cash_weight'])
        
        if len(portfolio_values) % 50 == 0:
            print(f"Day {len(portfolio_values)}: Value={info['portfolio_value']:,.0f} (Cash: {info['cash_weight']*100:.1f}%)")

    # ه) نتایج نهایی
    final_val = info['portfolio_value']
    initial_val = portfolio_values[0] if portfolio_values else 100_000_000
    roi = (final_val - initial_val) / initial_val * 100

    print("\n" + "="*50)
    print(f"🏁 Final Portfolio Value: {final_val:,.0f} Tomans")
    print(f"📈 Total Return (ROI): {roi:.2f}%")
    print("="*50)

    # و) رسم نمودار
    plt.figure(figsize=(12, 8))

    # 1. نمودار رشد سرمایه
    plt.subplot(2, 1, 1)
    plt.plot(date_list[:len(portfolio_values)], portfolio_values, label='AI Portfolio', color='blue', linewidth=2)
    plt.title(f'AI Performance (Forward Walk: {START_DATE_TEST} - Now)')
    plt.ylabel('Value (Tomans)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # خلوت کردن محور افقی
    plt.xticks(np.arange(0, len(date_list), step=max(1, len(date_list)//10)), rotation=45)

    # 2. نمودار مدیریت نقدینگی
    plt.subplot(2, 1, 2)
    plt.plot(date_list[:len(cash_ratios)], cash_ratios, label='Cash Allocation', color='green', alpha=0.7)
    plt.fill_between(range(len(cash_ratios)), cash_ratios, color='green', alpha=0.1)
    plt.title('Risk Management (Cash Position)')
    plt.ylabel('Cash Ratio (0-1)')
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(0, len(date_list), step=max(1, len(date_list)//10)), rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
