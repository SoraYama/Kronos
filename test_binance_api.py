#!/usr/bin/env python3
"""
币安API连接测试脚本
用于验证API连接和获取少量测试数据
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


def test_binance_api():
    """测试币安API连接"""

    print("测试币安API连接...")

    # 测试API连接
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            print("✅ 币安API连接正常")
        else:
            print(f"❌ API连接异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 网络连接失败: {e}")
        return False

    # 测试获取ETH/USDT最新价格
    try:
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ETH/USDT当前价格: ${data['price']}")
        else:
            print(f"❌ 获取价格失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取价格失败: {e}")
        return False

    # 测试获取少量历史数据
    try:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - 24 * 60 * 60 * 1000  # 24小时前

        params = {
            'symbol': 'ETHUSDT',
            'interval': '5m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 10
        }

        response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
        if response.status_code == 200:
            klines = response.json()
            print(f"✅ 成功获取 {len(klines)} 条历史数据")

            # 显示数据格式
            if klines:
                print("\n数据格式示例:")
                print("时间戳, 开盘价, 最高价, 最低价, 收盘价, 成交量, 成交额")
                for kline in klines[:3]:  # 显示前3条
                    timestamp = datetime.fromtimestamp(kline[0] / 1000)
                    print(f"{timestamp}, {kline[1]}, {kline[2]}, {kline[3]}, {kline[4]}, {kline[5]}, {kline[7]}")

            return True
        else:
            print(f"❌ 获取历史数据失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取历史数据失败: {e}")
        return False


def test_data_conversion():
    """测试数据格式转换"""

    print("\n测试数据格式转换...")

    # 模拟币安API返回的数据
    sample_klines = [
        [
            1690848000000,  # 时间戳
            "1850.50",      # 开盘价
            "1860.75",      # 最高价
            "1845.25",      # 最低价
            "1855.80",      # 收盘价
            "1234.56",      # 成交量
            1690848299999,  # 收盘时间
            "2291234.56",   # 成交额
            100,            # 成交笔数
            "600.00",       # 主动买入成交量
            "1112345.67",   # 主动买入成交额
            "0"             # 忽略
        ]
    ]

    # 转换为项目格式
    data = []
    for kline in sample_klines:
        data.append({
            'timestamps': datetime.fromtimestamp(kline[0] / 1000),
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5]),
            'amount': float(kline[7])
        })

    df = pd.DataFrame(data)

    print("✅ 数据格式转换成功")
    print("转换后的数据:")
    print(df)
    print(f"数据类型:")
    print(df.dtypes)

    return True


def main():
    """主函数"""
    print("=" * 50)
    print("币安API测试工具")
    print("=" * 50)

    # 测试API连接
    api_ok = test_binance_api()

    if api_ok:
        # 测试数据转换
        conversion_ok = test_data_conversion()

        if conversion_ok:
            print("\n🎉 所有测试通过！可以开始获取数据。")
            print("运行以下命令开始获取数据:")
            print("python simple_binance_fetcher.py")
        else:
            print("\n❌ 数据转换测试失败")
    else:
        print("\n❌ API连接测试失败，请检查网络连接")


if __name__ == "__main__":
    main()
