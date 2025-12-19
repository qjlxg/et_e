import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
import pytz
import logging
import math

# --- V5.0 策略所需配置参数 ---
FUND_DATA_DIR = 'fund_data'
MIN_MONTH_DRAWDOWN = 0.06 # V5.0 震荡市核心触发 (回撤 >= 6%)
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.15 # 高弹性策略的基础回撤要求 (15%)
MIN_DAILY_DROP_PERCENT = 0.03 # 当日大跌的定义 (3%)
REPORT_BASE_NAME = 'fund_warning_report_v5_merged_table'

# --- 核心阈值调整 ---
EXTREME_RSI_THRESHOLD_P1 = 29.0 # 网格级：RSI(14) 极值超卖
STRONG_RSI_THRESHOLD_P2 = 35.0 # 强力超卖观察池
SHORT_TERM_RSI_EXTREME = 20.0 # RSI(6)的极值超卖阈值
TREND_HEALTH_THRESHOLD = 0.9 # MA50/MA250 健康度阈值 (0.9)
MIN_BUY_SIGNAL_SCORE = 3.7 # 最低信号分数
TREND_SLOPE_THRESHOLD = 0.005 # 趋势拟合斜率阈值

# --- 设置日志 (1/15) ---
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fund_analysis.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# --- 数据预处理和验证 (2/15) ---
def load_and_preprocess_data(filepath, fund_code):
    """
    加载、预处理和验证基金数据。
    支持表头：date, net_value, cumulative_net_value, daily_growth_rate...
    """
    try:
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='gbk')
        
        # 统一列名映射逻辑 (针对用户提供的新表头)
        column_map = {
            'date': 'date',
            'net_value': 'value',
            'Date': 'date',
            'NetValue': 'value'
        }
        
        # 如果列名中存在 net_value，则将其重命名为分析用的 value
        current_cols = df.columns.tolist()
        rename_dict = {}
        for old_col, new_col in column_map.items():
            if old_col in current_cols:
                rename_dict[old_col] = new_col
        
        df = df.rename(columns=rename_dict)
        
        # 检查关键列
        if 'date' not in df.columns or 'value' not in df.columns:
            logging.warning(f"基金 {fund_code} 缺少 'date' 或 'net_value' 列。现有的列为: {df.columns.tolist()}")
            return None, "缺少关键列"
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
        
        if df.empty: return None, "数据为空"
        if len(df) < 60: return None, f"数据不足60条，当前只有{len(df)}条"
        if (df['value'] <= 0).any(): return None, "存在无效净值(<=0)"
        
        return df, "数据有效"
        
    except Exception as e:
        logging.error(f"加载基金 {fund_code} 数据时发生错误: {e}")
        return None, f"加载错误: {e}"

# --- 布林带计算 (3/15) ---
def calculate_bollinger_bands(series, window=20):
    """计算布林带位置"""
    if len(series) < window:
        return "数据不足"
    
    df_temp = pd.DataFrame({'value': series.values})
    df_temp['MA20'] = df_temp['value'].rolling(window=window).mean()
    df_temp['STD20'] = df_temp['value'].rolling(window=window).std()
    
    if df_temp['STD20'].iloc[-1] == 0:
        return "波动极小"
        
    df_temp['Upper Band'] = df_temp['MA20'] + (df_temp['STD20'] * 2)
    df_temp['Lower Band'] = df_temp['MA20'] - (df_temp['STD20'] * 2)
    
    latest_value = df_temp['value'].iloc[-1]
    latest_lower = df_temp['Lower Band'].iloc[-1]
    latest_upper = df_temp['Upper Band'].iloc[-1]
    
    if pd.isna(latest_lower) or pd.isna(latest_upper):
        return "数据不足"
        
    if latest_value <= latest_lower:
        return "**下轨下方**" 
    elif latest_value >= latest_upper:
        return "**上轨上方**" 
    else:
        range_band = latest_upper - latest_lower
        if range_band <= 1e-6:
            return "轨道中间" 
        position = (latest_value - latest_lower) / range_band
        if position < 0.2:
            return "下轨附近"
        elif position > 0.8:
            return "上轨附近"
        else:
            return "轨道中间"

# --- 技术指标计算 (4/15) ---
def calculate_technical_indicators(df):
    """计算基金净值的完整技术指标"""
    df_asc = df.copy()
    try:
        if 'value' not in df_asc.columns or len(df_asc) < 60:
             return {
                'RSI(14)': np.nan, 'RSI(6)': np.nan, 'MACD信号': '数据不足', 
                '净值/MA50': np.nan, '净值/MA250': np.nan, 'MA50/MA250': np.nan, 
                'MA50/MA250趋势': '数据不足', '布林带位置': '数据不足', 
                '最新净值': df_asc['value'].iloc[-1] if not df_asc.empty else np.nan,
                '当日跌幅': np.nan
             }

        delta = df_asc['value'].diff()
        for window in [14, 6]:
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(span=window, adjust=False, min_periods=1).mean()
            avg_loss = loss.ewm(span=window, adjust=False, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10) 
            df_asc[f'RSI_{window}'] = 100 - (100 / (1 + rs))

        rsi_14_latest = df_asc['RSI_14'].iloc[-1]
        rsi_6_latest = df_asc['RSI_6'].iloc[-1]
        
        ema_12 = df_asc['value'].ewm(span=12, adjust=False).mean()
        ema_26 = df_asc['value'].ewm(span=26, adjust=False).mean()
        df_asc['MACD'] = ema_12 - ema_26
        df_asc['Signal'] = df_asc['MACD'].ewm(span=9, adjust=False).mean()
        
        macd_latest = df_asc['MACD'].iloc[-1]
        signal_latest = df_asc['Signal'].iloc[-1]
        macd_prev = df_asc['MACD'].iloc[-2] if len(df_asc) >= 2 else np.nan
        signal_prev = df_asc['Signal'].iloc[-2] if len(df_asc) >= 2 else np.nan
        
        macd_signal = '观察'
        if not np.isnan(macd_prev) and not np.isnan(signal_prev):
            if macd_latest > signal_latest and macd_prev <= signal_prev:
                macd_signal = '强势金叉' if macd_latest > 0 else '弱势金叉'
            elif macd_latest < signal_latest and macd_prev >= signal_prev:
                macd_signal = '死叉' 
        
        df_asc['MA50'] = df_asc['value'].rolling(window=50, min_periods=1).mean()
        df_asc['MA250'] = df_asc['value'].rolling(window=250, min_periods=1).mean() 
        value_latest = df_asc['value'].iloc[-1]
        net_to_ma50 = value_latest / df_asc['MA50'].iloc[-1] if df_asc['MA50'].iloc[-1] != 0 else np.nan

        if len(df_asc) < 250:
            net_to_ma250, ma50_to_ma250, trend_direction = np.nan, np.nan, '数据不足'
        else:
            ma50_l, ma250_l = df_asc['MA50'].iloc[-1], df_asc['MA250'].iloc[-1]
            net_to_ma250 = value_latest / ma250_l if ma250_l != 0 else np.nan
            ma50_to_ma250 = ma50_l / ma250_l if ma250_l != 0 else np.nan
            recent_ratio = (df_asc['MA50'] / df_asc['MA250']).tail(50).dropna() 
            if len(recent_ratio) >= 5:
                slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
                trend_direction = '向上' if slope > TREND_SLOPE_THRESHOLD else ('向下' if slope < -TREND_SLOPE_THRESHOLD else '平稳')
            else: trend_direction = '数据不足'
        
        daily_drop = 0.0
        if len(df_asc) >= 2:
            v_prev = df_asc['value'].iloc[-2]
            if v_prev > 0: daily_drop = (value_latest - v_prev) / v_prev
            
        return {
            'RSI(14)': round(rsi_14_latest, 2) if not math.isnan(rsi_14_latest) else np.nan, 
            'RSI(6)': round(rsi_6_latest, 2) if not math.isnan(rsi_6_latest) else np.nan,     
            'MACD信号': macd_signal,
            '净值/MA50': round(net_to_ma50, 2) if not math.isnan(net_to_ma50) else np.nan,
            '净值/MA250': round(net_to_ma250, 2) if not math.isnan(net_to_ma250) else np.nan, 
            'MA50/MA250': round(ma50_to_ma250, 2) if not math.isnan(ma50_to_ma250) else np.nan, 
            'MA50/MA250趋势': trend_direction,
            '布林带位置': calculate_bollinger_bands(df_asc['value']), 
            '最新净值': round(value_latest, 4) if not math.isnan(value_latest) else np.nan,
            '当日跌幅': round(daily_drop, 4) 
        }
    except Exception as e:
        logging.error(f"技术指标错误: {e}")
        return {'RSI(14)': np.nan, 'MACD信号': '错误', '最新净值': np.nan, '当日跌幅': np.nan, 'MA50/MA250趋势': '错误', '布林带位置': '错误'}

# --- 连续下跌计算 (5/15) ---
def calculate_consecutive_drops(series):
    try:
        if series.empty or len(series) < 2: return 0
        drops = (series.diff() < 0).values
        count = 0
        for is_dropped in reversed(drops[1:]):
            if is_dropped: count += 1
            else: break
        return count
    except: return 0

# --- 最大回撤计算 (6/15) ---
def calculate_max_drawdown(series):
    try:
        if series.empty: return 0.0
        rolling_max = series.cummax()
        return ((rolling_max - series) / rolling_max).max()
    except: return 0.0

# --- 卖出/止损信号 (7/15) ---
def generate_exit_signal(row):
    rsi_14, macd, mdd = row.get('RSI(14)', np.nan), row.get('MACD信号', ''), row.get('最大回撤', 0.0)
    sigs = []
    if not pd.isna(rsi_14) and rsi_14 > 70.0: sigs.append("🚫 止盈：RSI(14) 过买")
    if macd == '死叉': sigs.append("🚫 止盈/止损：MACD死叉")
    if mdd > 0.10: sigs.append(f"🛑 止损：回撤超 10% ({mdd:.2%})")
    return ' | '.join(sigs) if sigs else "持有"

# --- V5.0 行动信号 (8/15) ---
def generate_v5_action_signal(row):
    rsi14, rsi6, macd, boll, mdd, drop, condrop = row.get('RSI(14)'), row.get('RSI(6)'), row.get('MACD信号'), row.get('布林带位置'), row.get('最大回撤', 0), row.get('当日跌幅', 0), row.get('近10日连跌', 0)
    sigs = []
    if not pd.isna(rsi14) and rsi14 <= EXTREME_RSI_THRESHOLD_P1:
        if rsi6 <= SHORT_TERM_RSI_EXTREME: sigs.append(f"💥【网格级】RSI极值共振(RSI14:{rsi14:.1f})")
        elif drop <= -MIN_DAILY_DROP_PERCENT: sigs.append(f"💥【网格级】RSI极值+恐慌(RSI14:{rsi14:.1f})")
        else: sigs.append(f"🌟【网格级】RSI极值(RSI14:{rsi14:.1f})")
    if mdd >= MIN_MONTH_DRAWDOWN:
        if condrop >= 5 and not any('网格级' in s for s in sigs): sigs.append("✨【震荡-连跌】连跌5日+高回撤") 
        if boll in ["**下轨下方**", "下轨附近"]: sigs.append("🎯【震荡-高吸】触及BOLL下轨")
        elif mdd >= HIGH_ELASTICITY_MIN_DRAWDOWN: sigs.append("🔥【震荡-预警】高弹性回撤达标")
        elif not sigs: sigs.append("【震荡-关注】基础回撤达标")
    if macd == '弱势金叉': sigs.append("🛡️【防御-反弹】MACD弱金叉")
    if not pd.isna(rsi14) and rsi14 > 70.0: sigs.append("🚫【牛市过滤器】RSI(14)>70")
    return ' | '.join(sigs) if sigs else '等待信号 (未达基础回撤)'

# --- 分析逻辑 (9-10/15) ---
def analyze_all_funds():
    files = glob.glob(os.path.join(FUND_DATA_DIR, '*.csv'))
    results = []
    for f in files:
        res = analyze_single_fund(f)
        if res: results.append(res)
    return results

def analyze_single_fund(filepath):
    code = os.path.splitext(os.path.basename(filepath))[0]
    df, msg = load_and_preprocess_data(filepath, code)
    if df is None: return None
    try:
        latest_date = df['date'].iloc[-1]
        df_recent = df[df['date'] >= (latest_date - pd.DateOffset(months=1))]['value']
        mdd = calculate_max_drawdown(df_recent) if len(df_recent) >= 2 else 0.0
        tech = calculate_technical_indicators(df)
        con_drop = calculate_consecutive_drops(df['value'].tail(10))
        row = {**tech, '最大回撤': mdd, '近10日连跌': con_drop}
        if not pd.isna(tech['最新净值']):
            return {'基金代码': code, '最大回撤': mdd, '最大连续下跌': calculate_consecutive_drops(df['value']), '近10日连跌': con_drop, **tech, '行动提示': generate_v5_action_signal(row), '退出提示': generate_exit_signal(row)}
        return None
    except: return None

# --- 格式化与报告 (11-14/15) ---
def format_technical_value(val, fmt='percent'):
    if pd.isna(val): return '---'
    if fmt == 'report_daily_drop': return f"**{val:.2%}**" if val < 0 else f"{val:.2%}"
    if fmt == 'percent': return f"{val:.2%}"
    return f"{val:.2f}"

def format_table_row(idx, row):
    trial_price = row.get('最新净值', 1.0) * 0.97
    trend, ratio = row['MA50/MA250趋势'], row.get('MA50/MA250')
    if pd.isna(ratio) or trend == '数据不足': ts = "---"
    elif trend == '向下' or ratio < TREND_HEALTH_THRESHOLD: ts = f"⚠️ **{trend}** ({ratio:.2f})"
    else: ts = f"**{trend}** ({ratio:.2f})"
    
    rsi_disp = f"**{row['RSI(14)']:.2f}**" if not pd.isna(row['RSI(14)']) and row['RSI(14)'] <= STRONG_RSI_THRESHOLD_P2 else f"{row['RSI(14)']:.2f}"
    v5_sig = f"🚫 **止损否决** | {row['行动提示']}" if "🛑 止损：" in row['退出提示'] else f"**{row['行动提示']}**"
    
    return (f"| {idx} | `{row['基金代码']}` | **{format_technical_value(row['最大回撤'], 'percent')}** | "
            f"{format_technical_value(row['当日跌幅'], 'report_daily_drop')} | {rsi_disp} | {v5_sig} | "
            f"**{row['退出提示']}** | {ts} | `{trial_price:.4f}` |\n")

def generate_merged_table(df_group):
    header = "| 排名 | 基金代码 | **最大回撤 (1M)** | **当日跌幅** | RSI(14) | **V5.0 信号** | **退出提示** | MA50/MA250健康度 | 试水买价 (跌3%) |\n"
    sep = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    parts = ["### 综合技术分析表\n", header, sep]
    for i, (_, row) in enumerate(df_group.iterrows(), 1):
        parts.append(format_table_row(i, row))
    parts.append("\n---\n")
    return "".join(parts)

def generate_report(results, ts_str):
    if not results: return f"# 基金预警报告 ({ts_str})\n\n**无数据**"
    df = pd.DataFrame(results)
    df_f = df[df['最大回撤'] >= MIN_MONTH_DRAWDOWN].copy()
    if df_f.empty: return f"# 基金报告 ({ts_str})\n\n**无触发回撤条件的基金**"

    # 评分逻辑
    score_map = {'💥【网格级】RSI极值共振': 5.0, '💥【网格级】RSI极值': 4.5, '🌟【网格级】RSI极值': 4.5, '🎯【震荡-高吸】': 4.0, '✨【震荡-连跌】': 3.5, '🛡️【防御-反弹】': 3.0, '🔥【震荡-预警】': 2.0, '【震荡-关注】': 1.0}
    df_f['signal_score'] = df_f['行动提示'].apply(lambda x: max([v for k, v in score_map.items() if k in x] + [0]))
    df_f['trend_score'] = df_f.apply(lambda r: 0 if r['MA50/MA250趋势'] == '向下' or r.get('MA50/MA250', 1) < TREND_HEALTH_THRESHOLD else 100, axis=1)
    df_f['is_stop_loss'] = np.where(df_f['最大回撤'] > 0.10, 1, 0)
    
    df_buy = df_f[(df_f['trend_score'] == 100) & (df_f['signal_score'] >= MIN_BUY_SIGNAL_SCORE)].sort_values(['is_stop_loss', 'signal_score', '最大回撤'], ascending=[True, False, False])
    df_i_buyable = df_buy[df_buy['is_stop_loss'] == 0]
    df_ii_rejected = df_buy[df_buy['is_stop_loss'] == 1]
    df_iv = df_f[df_f['trend_score'] == 0].sort_values(['最大回撤'], ascending=False)

    report = [f"# 基金 V5.0 策略报告 ({ts_str})\n\n", "## 分析总结\n\n", f"发现 **{len(df_f)}** 只基金入选。\n", f"**{len(df_i_buyable)}** 只可试仓。\n\n---\n"]
    if not df_i_buyable.empty:
        report.append(f"## 🥇 I.1 【最高优先级/可试仓】 ({len(df_i_buyable)}只)\n")
        report.append(generate_merged_table(df_i_buyable))
    if not df_ii_rejected.empty:
        report.append(f"## 🚫 I.2 【趋势健康但止损否决】 ({len(df_ii_rejected)}只)\n")
        report.append(generate_merged_table(df_ii_rejected))
    if not df_iv.empty:
        report.append(f"## ❌ IV. 【趋势不健康】 ({len(df_iv)}只)\n")
        report.append(generate_merged_table(df_iv))
    
    report.append("\n---\n## **✅ 核心决策纪律**\n1. 优先 I.1 组。\n2. 趋势向下必须放弃。\n")
    return "".join(report)

# --- 主函数 (15/15) ---
def main():
    setup_logging()
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    ts_file, ts_rep = now.strftime('%Y%m%d_%H%M%S'), now.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(now.strftime('%Y%m'), exist_ok=True)
    report_path = os.path.join(now.strftime('%Y%m'), f"{REPORT_BASE_NAME}_{ts_file}.md")
    
    if not os.path.isdir(FUND_DATA_DIR):
        os.makedirs(FUND_DATA_DIR, exist_ok=True)
        return False

    results = analyze_all_funds()
    content = generate_report(results, ts_rep)
    with open(report_path, 'w', encoding='utf-8') as f: f.write(content)
    logging.info(f"报告已生成: {report_path}")
    return True

if __name__ == '__main__':
    if main(): print("脚本执行完毕。已兼容新表头并更新报告。")
    else: print("执行失败，请检查日志。")