#!/usr/bin/env python3
"""
简化版币安数据获取脚本
专门用于获取ETH/USDT数据并转换为Kronos项目格式
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from tqdm import tqdm


def fetch_eth_usdt_data(start_date="2023-08-01", end_date="2025-09-30", interval="5m"):
    """
    获取ETH/USDT历史数据

    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        interval: 时间间隔 '5m', '1h', '1d' 等

    Returns:
        pandas.DataFrame: 包含OHLCV数据的DataFrame
    """

    # 币安API端点
    base_url = "https://api.binance.com/api/v3/klines"

    # 转换日期为时间戳
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    all_data = []
    current_start = start_ts

    # 计算总批次数（用于进度条）
    total_days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    if interval == "5m":
        batches = total_days * 24 * 12  # 每天288个5分钟K线
    elif interval == "1h":
        batches = total_days * 24
    elif interval == "1d":
        batches = total_days
    else:
        batches = total_days * 10  # 估算值

    print(f"开始获取ETH/USDT {interval}数据...")
    print(f"时间范围: {start_date} 到 {end_date}")
    print(f"预计需要 {batches//1000 + 1} 个批次")

    with tqdm(total=batches//1000 + 1, desc="获取数据") as pbar:
        while current_start < end_ts:
            # 设置请求参数
            params = {
                'symbol': 'ETHUSDT',
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': 1000
            }

            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    response = requests.get(base_url, params=params, timeout=30)
                    response.raise_for_status()
                    klines = response.json()

                    if not klines:
                        break

                    # 转换数据格式
                    for kline in klines:
                        all_data.append({
                            'timestamps': datetime.fromtimestamp(kline[0] / 1000),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'amount': float(kline[7])  # 成交额
                        })

                    # 更新开始时间
                    current_start = klines[-1][0] + 1
                    pbar.update(1)

                    # 避免请求过于频繁
                    time.sleep(0.2)
                    break  # 成功获取数据，跳出重试循环

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # 指数退避
                        print(f"\n请求失败 (尝试 {retry_count}/{max_retries}): {e}")
                        print(f"等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n请求失败，已达到最大重试次数: {e}")
                        print("跳过当前批次...")
                        break

    # 创建DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values('timestamps').reset_index(drop=True)

    return df


def save_and_validate_data(df, filename="ETHUSDT_5m_2023-08-01_to_2025-09-30.csv"):
    """保存数据并验证格式"""

    # 创建data目录
    data_dir = "examples/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filepath = os.path.join(data_dir, filename)

    # 保存数据
    df.to_csv(filepath, index=False)

    print(f"\n数据已保存到: {filepath}")
    print(f"数据条数: {len(df):,}")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")

    # 验证数据格式
    print(f"\n数据格式验证:")
    print(f"列名: {list(df.columns)}")
    print(f"数据类型:")
    print(df.dtypes)

    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\n警告: 发现缺失值:")
        print(missing_values[missing_values > 0])
    else:
        print("\n✓ 无缺失值")

    # 显示数据预览
    print(f"\n数据预览:")
    print(df.head())

    return filepath


def main():
    """主函数"""
    print("=" * 60)
    print("币安ETH/USDT数据获取工具")
    print("=" * 60)

    # 获取数据
    df = fetch_eth_usdt_data(
        start_date="2023-08-01",
        end_date="2025-09-30",
        interval="5m"
    )

    if df.empty:
        print("❌ 未获取到数据")
        return

    # 保存和验证数据
    filepath = save_and_validate_data(df)

    print(f"\n✅ 数据获取完成!")
    print(f"文件位置: {filepath}")
    print(f"数据可用于Kronos模型训练和预测")


if __name__ == "__main__":
    main()
