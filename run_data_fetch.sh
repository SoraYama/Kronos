#!/bin/bash

echo "=========================================="
echo "币安ETH/USDT数据获取工具"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
python3 -c "import requests, pandas, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖包，正在安装..."
    pip3 install requests pandas tqdm
fi

# 运行API测试
echo "运行API连接测试..."
python3 test_binance_api.py

if [ $? -eq 0 ]; then
    echo ""
    echo "开始获取数据..."
    python3 simple_binance_fetcher.py
else
    echo "❌ API测试失败，请检查网络连接"
    exit 1
fi
