#data_loader.py
import pandas as pd
import numpy as np
import pytse_client as tse
import os
import pickle

def fetch_and_clean_data(tickers, save_path="clean_market_data.pkl", force_update=False):
    """
    نسخه اصلاح شده: استفاده از دانلود دسته‌جمعی (Batch) برای رفع باگ کتابخانه.
    """
    
    # 1. اگر فایل هست و اجبار به آپدیت نداریم، لود کن
    if os.path.exists(save_path) and not force_update:
        print(f"📂 Loading existing data from {save_path}...")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"⏳ Starting BATCH download for {len(tickers)} tickers...")
    
    # 2. دانلود یکباره (Batch Download) - رفع باگ unhashable type
    try:
        # کل لیست را یکجا می‌دهیم. خروجی یک دیکشنری است: { 'namad': DataFrame, ... }
        raw_data_map = tse.download(tickers, write_to_csv=False, adjust=True)
    except Exception as e:
        print(f"❌ Critical Error in Batch Download: {e}")
        return None, None

    # بررسی اینکه آیا دیتایی آمد یا نه
    if not raw_data_map:
        print("❌ No data received from TSE Client.")
        return None, None

    print(f"   ✅ Download successful. Received {len(raw_data_map)} tickers.")
    
    data_map = {}
    valid_tickers = []

    # 3. پیش‌پردازش اولیه روی هر سهم
    for ticker in tickers:
        if ticker not in raw_data_map:
            print(f"   ⚠️ Warning: Data for {ticker} not returned by server. Skipping.")
            continue
            
        df = raw_data_map[ticker]
        
        # بررسی خالی نبودن
        if df.empty:
            print(f"   ⚠️ Warning: {ticker} has empty data. Skipping.")
            continue

        try:
            # استانداردسازی ستون‌ها
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            data_map[ticker] = df
            valid_tickers.append(ticker)
            # print(f"      -> {ticker}: {len(df)} days.")
        except KeyError as e:
            print(f"   ❌ Data format error for {ticker}: Missing column {e}")

    if not valid_tickers:
        print("❌ No valid data available after processing.")
        return None, None

    # 4. همگام‌سازی زمانی (The Synchronization)
    # پیدا کردن تاریخی که همه سهم‌های معتبر در آن حضور دارند (ماکسیممِ تاریخ‌های شروع)
    start_dates = [data_map[t].index.min() for t in valid_tickers]
    common_start_date = max(start_dates)
    
    end_dates = [data_map[t].index.max() for t in valid_tickers]
    common_end_date = min(end_dates)

    print(f"\n📅 Synchronization Info:")
    print(f"   Common Start Date: {common_start_date.date()}")
    print(f"   Common End Date:   {common_end_date.date()}")
    
    if common_start_date >= common_end_date:
        print("❌ Date overlap issue: Start date is after End date. Check your tickers list.")
        return None, None

    # تقویم مرجع
    full_date_range = pd.date_range(start=common_start_date, end=common_end_date, freq='D')
    print(f"   Total Days: {len(full_date_range)}")

    # 5. ساخت ماتریس نهایی
    n_days = len(full_date_range)
    n_assets = len(valid_tickers)
    n_features = 5
    
    final_data = np.zeros((n_days, n_assets, n_features))
    
    print("\n🔄 Filling Gaps (FFILL)...")
    for i, ticker in enumerate(valid_tickers):
        df = data_map[ticker]
        
        # بازچینی و پر کردن جاهای خالی با قیمت روز قبل
        df_reindexed = df.reindex(full_date_range)
        df_filled = df_reindexed.ffill() 
        df_filled = df_filled.bfill() # برای محکم کاری روز اول
        df_filled = df_filled.fillna(0) # نباید اتفاق بیفتد

        final_data[:, i, 0] = df_filled['open'].values
        final_data[:, i, 1] = df_filled['high'].values
        final_data[:, i, 2] = df_filled['low'].values
        final_data[:, i, 3] = df_filled['close'].values
        final_data[:, i, 4] = df_filled['volume'].values

    # 6. چک نهایی
    if np.min(final_data[:, :, 3]) < 10.0:
        print("⚠️ Warning: Still found prices < 10.0. Inspect data carefully.")
    else:
        print("✅ Data Integrity Check Passed (No zero prices).")

    final_dates = full_date_range.strftime("%Y-%m-%d").tolist()
    
    # ذخیره
    with open(save_path, 'wb') as f:
        pickle.dump((final_data, final_dates, valid_tickers), f) # valid_tickers را هم ذخیره می‌کنیم
        
    print(f"💾 Saved to {save_path}")
    
    # نکته: خروجی valid_tickers را هم برمی‌گردانیم شاید لیست اولیه تغییر کرده باشد
    return final_data, final_dates, valid_tickers

if __name__ == "__main__":
    # تست
    my_tickers = ["فولاد", "شپنا", "وبملت", "فملی", "شستا", "خودرو", "وتجارت"]
    # my_tickers = ["فولاد", "شپنا"] # تست با تعداد کم
    d, dates, final_tickers = fetch_and_clean_data(my_tickers, force_update=True)
    
    if d is not None:
        print("Output Shape:", d.shape)
        print("Final Tickers:", final_tickers)
