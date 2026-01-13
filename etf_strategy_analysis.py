import os
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 战法名称：【底部长城 · 超跌放量反弹战法】
# ==========================================
# 逻辑说明：
# 1. 寻找近期（1个月）跌幅较大，且处于连跌状态的品种。
# 2. 核心逻辑：RSI < 30 进入超卖区，寻找“缩量下跌后放量止跌”信号。
# 3. 买入点：RSI底背离或连跌后收出第一根放量阳线。
# 4. 卖出点：RSI > 70 或短期反弹超过 8-10%。
# 
# 操作要领：
# - 优先看“止跌放量”标签为“是”的品种。
# - 振幅大的品种适合配合日内分时低吸做T，降低持仓成本。
# ==========================================

def calculate_rsi(series, period=14):
    if len(series) < period: return pd.Series([np.nan] * len(series))
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_single_file(file_path):
    try:
        symbol = os.path.basename(file_path).replace('.csv', '')
        # 使用 utf-8-sig 处理 BOM 头
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.lower() # 统一转小写去空格
        
        # 兼容性映射：中文列名映射到统一键名
        col_map = {
            '日期': 'date', 'date': 'date',
            '收盘': 'close', 'net_value': 'close',
            '成交量': 'vol', 'volume': 'vol',
            '振幅': 'amp', 'amplitude': 'amp',
            '开盘': 'open', 'open': 'open',
            '涨跌额': 'chg_amt'
        }
        
        # 重命名现有的列
        current_cols = {c: col_map[c] for c in df.columns if c in col_map}
        df = df.rename(columns=current_cols)

        if 'date' not in df.columns or 'close' not in df.columns:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if len(df) < 20: return None

        # --- 核心指标计算 ---
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        last_close = last_row['close']
        
        # 1. 跌幅统计 (1个月约20交易日)
        pct_1m = round((last_close - df['close'].iloc[max(-len(df), -21)]) / df['close'].iloc[max(-len(df), -21)] * 100, 2)

        # 2. 振幅 (做T参考)
        avg_amp_10d = round(df['amp'].tail(10).mean(), 2) if 'amp' in df.columns else 0
        
        # 3. 连跌天数 (基于收盘价)
        close_values = df['close'].values
        consecutive_drops = 0
        for i in range(len(close_values)-1, 0, -1):
            if close_values[i] < close_values[i-1]:
                consecutive_drops += 1
            else:
                break
        
        # 4. RSI 指标
        df['rsi'] = calculate_rsi(df['close'])
        current_rsi = round(df['rsi'].iloc[-1], 2)
        
        # 5. 放量止跌信号 (今日收阳 且 相比昨日放量20%)
        is_stop_drop = False
        if 'open' in df.columns and 'vol' in df.columns:
            is_stop_drop = (last_row['close'] > last_row['open']) and (last_row['vol'] > prev_row['vol'] * 1.2)

        # --- 筛选逻辑：超跌 或 RSI超卖 ---
        if current_rsi < 35 or (consecutive_drops >= 3 and pct_1m < -5):
            return {
                "证券代码": symbol,
                "当前价": last_close,
                "RSI(14)": current_rsi,
                "连跌天数": consecutive_drops,
                "1月跌幅%": pct_1m,
                "10日均振幅%": avg_amp_10d,
                "止跌放量": "是" if is_stop_drop else "否",
                "战法建议": "超卖反弹" if current_rsi < 30 else "等待放量"
            }
    except Exception as e:
        print(f"解析 {file_path} 失败: {e}")
    return None

def main():
    # 自动识别 fund_data 或 stock_data 目录
    data_dir = 'stock_data' if os.path.exists('stock_data') else 'fund_data'
    etf_list_path = 'ETF列表.xlsx - Sheet1.csv'
    
    if not os.path.exists(data_dir):
        print(f"错误: 未找到数据目录 {data_dir}")
        return

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(analyze_single_file, files))
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("今日无符合战法条件的品种")
        return

    result_df = pd.DataFrame(valid_results)
    
    # 匹配 ETF 名称
    if os.path.exists(etf_list_path):
        name_df = pd.read_csv(etf_list_path, encoding='utf-8-sig')
        name_df.columns = name_df.columns.str.strip()
        name_df['证券代码'] = name_df['证券代码'].astype(str).str.zfill(6)
        result_df['证券代码'] = result_df['证券代码'].astype(str).str.zfill(6)
        result_df = pd.merge(result_df, name_df[['证券代码', '证券简称']], on='证券代码', how='left')

    # 排序：按RSI升序 (越低越超跌)
    result_df = result_df.sort_values(by="RSI(14)")

    # 保存结果到年月目录
    now = datetime.now()
    dir_path = now.strftime('%Y/%m')
    os.makedirs(dir_path, exist_ok=True)
    
    timestamp = now.strftime('%Y%m%d_%H%M')
    file_name = f"etf_rebound_strategy_{timestamp}.csv"
    save_path = os.path.join(dir_path, file_name)
    
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"分析完成！结果已存入: {save_path}")

if __name__ == "__main__":
    main()
