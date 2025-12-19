import pandas as pd
import requests
import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import re
from datetime import datetime
import logging
import concurrent.futures
from functools import partial

# ================= 配置区 =================
# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = 'C类.txt'        # 输入基金代码文件
OUTPUT_DIR = 'fund_data'     # 输出文件夹
# 天天基金历史净值 API 地址
BASE_URL_NET_VALUE = "http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page={page_index}&per=20"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'http://fund.eastmoney.com/',
}

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 3.5          # 请求页面的基础延迟
MAX_CONCURRENT = 15          # 最大并发抓取基金数
MAX_FUNDS_PER_RUN = 0        # 限制运行数量，0表示抓取全部
# ==========================================

def get_all_fund_codes(file_path):
    """从本地文件读取 6 位基金代码"""
    logger.info(f"正在读取代码文件: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"文件 {file_path} 不存在")
        return []

    try:
        # 尝试常用编码读取
        for enc in ['utf-8', 'gbk', 'utf-8-sig']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        codes = [line.strip() for line in lines if line.strip()]
        valid_codes = [c for c in codes if re.match(r'^\d{6}$', c)]
        logger.info(f"成功获取 {len(valid_codes)} 个有效基金代码")
        return valid_codes
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return []

def load_latest_date(fund_code):
    """读取本地 CSV 以获取增量抓取的起始日期"""
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if os.path.exists(output_path):
        try:
            # 仅读取 date 列以提高速度
            df = pd.read_csv(output_path, usecols=['date'], parse_dates=['date'], encoding='utf-8')
            if not df.empty:
                return df['date'].max().to_pydatetime().date()
        except Exception:
            return None
    return None

async def fetch_page(session, url):
    """执行异步 HTTP GET 请求"""
    async with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT) as response:
        if response.status == 200:
            return await response.text()
        elif response.status == 514:
            raise aiohttp.ClientError("触发频率限制 (514)")
        else:
            raise aiohttp.ClientError(f"HTTP 错误: {response.status}")

async def fetch_net_values(fund_code, session, semaphore, executor):
    """核心抓取函数：支持 ETF 增量逻辑"""
    async with semaphore:
        logger.info(f"开始处理基金: {fund_code}")
        all_records = []
        
        # 线程池异步读取本地最新日期
        latest_date = await asyncio.get_event_loop().run_in_executor(executor, load_latest_date, fund_code)
        
        try:
            # 1. 抓取第一页获取总信息
            url_p1 = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=1)
            text = await fetch_page(session, url_p1)
            
            # 解析总页数
            total_pages_match = re.search(r'pages:(\d+)', text)
            total_pages = int(total_pages_match.group(1)) if total_pages_match else 1
            
            soup = BeautifulSoup(text, 'lxml')
            table = soup.find('table')
            if not table:
                return fund_code, "无数据表"
                
            rows = table.find_all('tr')[1:] # 跳过表头
            if not rows:
                return fund_code, "记录为空"

            # 检查 API 最新日期
            api_latest_str = rows[0].find_all('td')[0].text.strip()
            api_latest_date = datetime.strptime(api_latest_str, '%Y-%m-%d').date()

            # 如果本地已是最新，直接跳过
            if latest_date and api_latest_date <= latest_date:
                return fund_code, f"已是最新 ({latest_date})"

            # 2. 循环抓取各页
            page_index = 1
            stop_fetch = False
            while page_index <= total_pages and not stop_fetch:
                if page_index > 1:
                    await asyncio.sleep(REQUEST_DELAY)
                    current_url = BASE_URL_NET_VALUE.format(fund_code=fund_code, page_index=page_index)
                    text = await fetch_page(session, current_url)
                    rows = BeautifulSoup(text, 'lxml').find('table').find_all('tr')[1:]

                for row in rows:
                    tds = row.find_all('td')
                    if len(tds) < 7: continue
                    
                    date_str = tds[0].text.strip()
                    row_date = datetime.strptime(date_str, '%Y-%m-%d').date()

                    # 增量终止条件
                    if latest_date and row_date <= latest_date:
                        stop_fetch = True
                        break

                    # 收集数据项
                    all_records.append({
                        'date': date_str,
                        'net_value': tds[1].text.strip(),
                        'cumulative_net_value': tds[2].text.strip(),
                        'daily_growth_rate': tds[3].text.strip(),
                        'purchase_status': tds[4].text.strip(),
                        'redemption_status': tds[5].text.strip(),
                        'dividend': tds[6].text.strip() if len(tds) > 6 else ''
                    })
                page_index += 1

            return fund_code, all_records

        except Exception as e:
            logger.error(f"抓取基金 {fund_code} 异常: {e}")
            return fund_code, str(e)

def save_to_csv(fund_code, data):
    """
    修改后的保存逻辑：针对 ETF 格式进行适配
    格式：date, net_value, cumulative_net_value, daily_growth_rate, purchase_status, redemption_status, dividend
    """
    output_path = os.path.join(OUTPUT_DIR, f"{fund_code}.csv")
    if not isinstance(data, list) or not data:
        return False, 0

    try:
        new_df = pd.DataFrame(data)
        
        # 1. 净值转数值
        new_df['net_value'] = pd.to_numeric(new_df['net_value'], errors='coerce').round(4)
        new_df['cumulative_net_value'] = pd.to_numeric(new_df['cumulative_net_value'], errors='coerce').round(4)
        
        # 2. 增长率处理 (核心修改：ETF 经常返回 1.2% 这种格式)
        def parse_rate(val):
            val = str(val).strip()
            if val in ['--', '', 'nan']: return 0.0
            try:
                if '%' in val:
                    return float(val.replace('%', '')) / 100.0
                return float(val)
            except: return 0.0
        
        new_df['daily_growth_rate'] = new_df['daily_growth_rate'].apply(parse_rate)
        
        # 3. 日期转换与去空
        new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
        new_df.dropna(subset=['date'], inplace=True)

        # 4. 读取旧数据合并
        old_count = 0
        if os.path.exists(output_path):
            old_df = pd.read_csv(output_path, parse_dates=['date'], encoding='utf-8')
            old_count = len(old_df)
            combined = pd.concat([new_df, old_df], ignore_index=True)
        else:
            combined = new_df

        # 5. 去重、排序、格式化日期
        combined.drop_duplicates(subset=['date'], keep='first', inplace=True)
        combined.sort_values(by='date', ascending=False, inplace=True)
        combined['date'] = combined['date'].dt.strftime('%Y-%m-%d')

        # 6. 强制列顺序 (确保满足用户要求的 CSV 结构)
        target_cols = ['date', 'net_value', 'cumulative_net_value', 'daily_growth_rate', 
                       'purchase_status', 'redemption_status', 'dividend']
        
        for col in target_cols:
            if col not in combined.columns:
                combined[col] = "" # 缺失列补全
        
        final_df = combined[target_cols]

        # 7. 写入文件
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        
        added = len(final_df) - old_count
        logger.info(f"基金 {fund_code} 保存成功: 新增 {max(0, added)} 条记录")
        return True, max(0, added)

    except Exception as e:
        logger.error(f"保存基金 {fund_code} 失败: {e}")
        return False, 0

async def fetch_all_funds(fund_codes):
    """调度所有抓取任务"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    loop = asyncio.get_event_loop()
    
    # 线程池用于处理文件 I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
        async with ClientSession(connector=connector) as session:
            
            tasks = [fetch_net_values(f, session, semaphore, executor) for f in fund_codes]
            
            success_count = 0
            total_added = 0
            failed_list = []
            
            # 包装保存函数为异步可执行
            save_fn = partial(loop.run_in_executor, executor, save_to_csv)

            for future in asyncio.as_completed(tasks):
                fund_code, result = await future
                
                if isinstance(result, list): # 抓取成功，返回的是列表
                    ok, count = await save_fn(fund_code, result)
                    if ok:
                        success_count += 1
                        total_added += count
                    else:
                        failed_list.append(fund_code)
                else:
                    # 如果不是列表，可能是“已最新”或“错误”
                    if "已是最新" not in str(result):
                        failed_list.append(fund_code)
            
            return success_count, total_added, failed_list

def main():
    """入口函数"""
    print("="*40)
    print("基金/ETF 净值抓取程序 (增量模式)")
    print("="*40)
    
    # 初始化环境
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取代码
    codes = get_all_fund_codes(INPUT_FILE)
    if not codes:
        print("未找到待抓取的基金代码，请检查 C类.txt")
        return

    # 限制处理数量
    target_codes = codes[:MAX_FUNDS_PER_RUN] if MAX_FUNDS_PER_RUN > 0 else codes
    
    # 启动事件循环
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    start_time = time.time()
    s_count, t_added, f_list = asyncio.run(fetch_all_funds(target_codes))
    
    # 总结
    duration = time.time() - start_time
    print("\n" + "="*40)
    print(f"抓取结束！总耗时: {duration:.2f} 秒")
    print(f"成功处理: {s_count} 个基金")
    print(f"新增记录: {t_added} 条")
    if f_list:
        print(f"失败列表: {', '.join(f_list)}")
    print("="*40)

if __name__ == "__main__":
    main()