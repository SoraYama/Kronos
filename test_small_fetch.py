#!/usr/bin/env python3
"""
小规模数据获取测试
只获取最近7天的数据来测试功能
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os


def fetch_small_test_data():
    """获取少量测试数据"""

    print("获取ETH/USDT最近7天的5分钟数据...")

    # 币安API端点
    base_url = "https://api.binance.com/api/v3/klines"

    # 获取最近7天的数据
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - 7 * 24 * 60 * 60 * 1000  # 7天前

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            'symbol': 'ETHUSDT',
            'interval': '5m',
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000
        }

        try:
            print(f"请求数据: {datetime.fromtimestamp(current_start/1000)}")
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
                    'amount': float(kline[7])
                })

            # 更新开始时间
            current_start = klines[-1][0] + 1
            print(f"获取到 {len(klines)} 条数据")

            # 避免请求过于频繁
            time.sleep(0.5)

        except Exception as e:
            print(f"请求失败: {e}")
            time.sleep(2)
            continue

    # 创建DataFrame
    df = pd.DataFrame(all_data)
    df = df.sort_values('timestamps').reset_index(drop=True)

    return df


def main():
    """主函数"""
    print("=" * 50)
    print("ETH/USDT小规模数据获取测试")
    print("=" * 50)

    # 获取测试数据
    df = fetch_small_test_data()

    if df.empty:
        print("❌ 未获取到数据")
        return

    # 保存数据
    data_dir = "examples/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = "ETHUSDT_5m_test_7days.csv"
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"\n✅ 测试数据获取完成!")
    print(f"数据条数: {len(df):,}")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
    print(f"保存位置: {filepath}")

    # 显示数据预览
    print(f"\n数据预览:")
    print(df.head())

    print(f"\n数据统计:")
    print(df.describe())


if __name__ == "__main__":
    main()
