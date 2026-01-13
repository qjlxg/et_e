import os
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 战法名称：【底部长城 · 超跌放量反弹战法】
# 操作要领：
# 1. 寻找近期（1个月）跌幅较大，且处于连跌状态的品种。
# 2. 核心逻辑：RSI < 30 进入超卖区，寻找“缩量下跌后放量止跌”的信号。
# 3. 买入点：RSI底背离或连跌后收出第一根放量阳线。
# 4. 卖出点：RSI > 70 或短期反弹超过 8-10%。
# ==========================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_single_file(file_path):
    try:
        symbol = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        
        # 统一日期格式并排序
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        
        if len(df) < 30: return None

        # 基础指标计算
        last_close = df['收盘'].iloc[-1]
        
        # 1. 计算涨跌幅统计
        def get_pct_chg(days):
            if len(df) < days: return np.nan
            start_price = df['收盘'].iloc[-days]
            return round((last_close - start_price) / start_price * 100, 2)

        pct_1w = get_pct_chg(5)
        pct_2w = get_pct_chg(10)
        pct_1m = get_pct_chg(20)

        # 2. 计算平均振幅 (近10日)
        avg_amplitude = round(df['振幅'].tail(10).mean(), 2)

        # 3. 计算连跌天数
        change = df['涨跌额'].values
        consecutive_drops = 0
        for i in range(len(change)-1, -1, -1):
            if change[i] < 0:
                consecutive_drops += 1
            else:
                break
        
        # 4. 计算 RSI (14)
        df['RSI'] = calculate_rsi(df['收盘'])
        current_rsi = round(df['RSI'].iloc[-1], 2)

        # --- 筛选逻辑 (示例条件) ---
        # 条件：1个月跌幅 > 5% 且 RSI < 40 (接近超卖) 或 连跌3天以上
        is_candidate = (pct_1m < -5 and current_rsi < 40) or (consecutive_drops >= 3)

        if is_candidate:
            return {
                "证券代码": symbol,
                "当前收盘": last_close,
                "一周涨跌%": pct_1w,
                "半月涨跌%": pct_2w,
                "一月涨跌%": pct_1m,
                "10日均振幅": avg_amplitude,
                "连跌天数": consecutive_drops,
                "RSI(14)": current_rsi
            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def main():
    data_dir = 'stock_data'
    etf_list_path = 'ETF列表.xlsx - Sheet1.csv' # 根据你上传的文件名
    
    # 1. 获取所有待处理文件
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 2. 多进程并行计算以加快速度
    cpu_count = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = list(executor.map(analyze_single_file, files))
    
    # 3. 汇总结果
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("今日无符合条件的筛选结果")
        return

    result_df = pd.DataFrame(valid_results)
    
    # 4. 匹配 ETF 名称
    if os.path.exists(etf_list_path):
        name_df = pd.read_csv(etf_list_path)
        name_df['证券代码'] = name_df['证券代码'].astype(str).str.zfill(6)
        result_df['证券代码'] = result_df['证券代码'].astype(str).str.zfill(6)
        result_df = pd.merge(result_df, name_df, on='证券代码', how='left')

    # 5. 保存结果
    now = datetime.now()
    dir_path = now.strftime('%Y/%m')
    os.makedirs(dir_path, exist_ok=True)
    
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    file_name = f"etf_strategy_analysis_{timestamp}.csv"
    save_path = os.path.join(dir_path, file_name)
    
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"分析完成，结果保存至: {save_path}")

if __name__ == "__main__":
    main()
