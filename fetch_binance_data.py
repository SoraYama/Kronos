#!/usr/bin/env python3
"""
币安ETH/USDT数据获取脚本
从币安API获取ETH/USDT的历史K线数据并转换为项目所需格式
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import json
from typing import List, Dict, Optional


class BinanceDataFetcher:
    """币安数据获取器"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.session = requests.Session()
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List[List]:
        """
        获取K线数据

        Args:
            symbol: 交易对，如 'ETHUSDT'
            interval: 时间间隔，如 '5m', '1h', '1d'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            limit: 每次请求的最大条数（最大1000）

        Returns:
            K线数据列表
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return []

    def fetch_historical_data(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史数据

        Args:
            symbol: 交易对
            interval: 时间间隔
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            包含历史数据的DataFrame
        """
        # 转换日期为时间戳
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        all_data = []
        current_start = start_timestamp

        print(f"开始获取 {symbol} 从 {start_date} 到 {end_date} 的 {interval} 数据...")

        while current_start < end_timestamp:
            # 计算当前批次的结束时间
            current_end = min(current_start + 1000 * self._interval_to_ms(interval), end_timestamp)

            print(f"获取数据: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")

            klines = self.get_klines(symbol, interval, current_start, current_end)

            if not klines:
                print("获取数据失败，等待5秒后重试...")
                time.sleep(5)
                continue

            all_data.extend(klines)

            # 更新开始时间为最后一条数据的时间 + 1个间隔
            if klines:
                last_timestamp = klines[-1][0]
                current_start = last_timestamp + self._interval_to_ms(interval)
            else:
                break

            # 避免请求过于频繁
            time.sleep(0.1)

        return self._convert_to_dataframe(all_data)

    def _interval_to_ms(self, interval: str) -> int:
        """将时间间隔转换为毫秒"""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return interval_map.get(interval, 60 * 1000)

    def _convert_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        将币安K线数据转换为项目所需格式

        币安K线数据格式:
        [
            [
                1499040000000,      // 开盘时间
                "0.01634790",       // 开盘价
                "0.80000000",       // 最高价
                "0.01575800",       // 最低价
                "0.01577100",       // 收盘价
                "148976.11427815",  // 成交量
                1499644799999,      // 收盘时间
                "2434.19055334",    // 成交额
                308,                // 成交笔数
                "1756.87402397",    // 主动买入成交量
                "28.46694368",      // 主动买入成交额
                "17928899.62484339" // 忽略此参数
            ]
        ]
        """
        if not klines:
            return pd.DataFrame()

        data = []
        for kline in klines:
            data.append({
                'timestamps': datetime.fromtimestamp(kline[0] / 1000),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'amount': float(kline[7])  # 成交额
            })

        df = pd.DataFrame(data)
        df = df.sort_values('timestamps').reset_index(drop=True)
        return df

    def save_data(self, df: pd.DataFrame, filename: str, output_dir: str = "data"):
        """保存数据到CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"数据已保存到: {filepath}")
        print(f"数据条数: {len(df)}")
        print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")


def main():
    """主函数"""
    fetcher = BinanceDataFetcher()

    # 配置参数
    symbol = "ETHUSDT"
    interval = "5m"  # 5分钟K线
    start_date = "2023-08-01"
    end_date = "2025-09-30"

    # 获取数据
    df = fetcher.fetch_historical_data(symbol, interval, start_date, end_date)

    if df.empty:
        print("未获取到数据")
        return

    # 保存数据
    filename = f"{symbol}_{interval}_{start_date}_to_{end_date}.csv"
    fetcher.save_data(df, filename)

    # 显示数据预览
    print("\n数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())


if __name__ == "__main__":
    main()
