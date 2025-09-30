# Kronos 加密货币预测使用指南

## 📋 目录
1. [环境安装和项目启动](#1-环境安装和项目启动)
2. [加密货币数据准备](#2-加密货币数据准备)
3. [执行预测命令](#3-执行预测命令)
4. [Web UI 使用](#4-web-ui-使用)
5. [高级功能](#5-高级功能)
6. [常见问题](#6-常见问题)

---

## 1. 环境安装和项目启动

### 1.1 系统要求
- **Python**: 3.10+
- **操作系统**: Windows/macOS/Linux
- **内存**: 建议 8GB+
- **GPU**: 可选，支持 CUDA/MPS 加速

### 1.2 安装步骤

#### 方法一：完整安装（推荐）
```bash
# 1. 克隆项目
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos

# 2. 安装核心依赖
pip install -r requirements.txt

# 3. 安装 Web UI 依赖
cd webui
pip install -r requirements.txt
cd ..
```

#### 方法二：仅安装核心功能
```bash
# 安装核心依赖
pip install numpy pandas torch einops huggingface_hub matplotlib tqdm safetensors
```

### 1.3 启动项目

#### 启动 Web UI（推荐新手）
```bash
# 方法1：使用启动脚本
cd webui
python run.py

# 方法2：使用 Shell 脚本（Linux/macOS）
cd webui
chmod +x start.sh
./start.sh

# 方法3：直接启动
cd webui
python app.py
```

启动成功后访问：http://localhost:7070

#### 启动命令行预测
```bash
# 进入项目根目录
cd Kronos

# 运行预测示例
python examples/prediction_example.py
```

---

## 2. 加密货币数据准备

### 2.1 数据格式要求

#### 必需列（Required Columns）
```csv
timestamps,open,high,low,close,volume
2024-01-01 00:00:00,42000.0,42500.0,41800.0,42200.0,1500.5
2024-01-01 01:00:00,42200.0,42800.0,42100.0,42600.0,1800.2
```

#### 可选列（Optional Columns）
- `amount`: 交易金额（如果不提供，会自动计算）
- `timestamp` 或 `date`: 时间戳的替代列名

### 2.2 数据获取方式

#### 从交易所 API 获取
```python
import pandas as pd
import ccxt

# 使用 Binance API 获取 BTC/USDT 数据
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'  # 1小时K线
limit = 1000      # 获取1000根K线

# 获取历史数据
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# 转换为 DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.drop('timestamp', axis=1)

# 保存数据
df.to_csv('btc_usdt_1h.csv', index=False)
```

#### 从数据提供商获取
```python
# 使用 yfinance 获取加密货币数据
import yfinance as yf

# 获取 BTC-USD 数据
btc = yf.Ticker("BTC-USD")
df = btc.history(period="1y", interval="1h")

# 重命名列以匹配 Kronos 格式
df = df.reset_index()
df.columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']]

# 保存数据
df.to_csv('btc_usd_1h.csv', index=False)
```

### 2.3 数据预处理

#### 数据清洗脚本
```python
import pandas as pd
import numpy as np

def preprocess_crypto_data(file_path, output_path):
    """
    预处理加密货币数据
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 处理时间戳
    if 'timestamp' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)
    elif 'date' in df.columns:
        df['timestamps'] = pd.to_datetime(df['date'])
        df = df.drop('date', axis=1)

    # 确保数值列为浮点数
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # 删除包含 NaN 的行
    df = df.dropna()

    # 按时间排序
    df = df.sort_values('timestamps').reset_index(drop=True)

    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"数据预处理完成，共 {len(df)} 条记录")
    return df

# 使用示例
df = preprocess_crypto_data('raw_btc_data.csv', 'btc_processed.csv')
```

### 2.4 数据质量检查

```python
def validate_crypto_data(df):
    """
    验证加密货币数据质量
    """
    print("=== 数据质量检查 ===")
    print(f"数据行数: {len(df)}")
    print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")

    # 检查必需列
    required_cols = ['timestamps', 'open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False

    # 检查价格逻辑
    invalid_price = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
    if invalid_price.any():
        print(f"❌ 发现 {invalid_price.sum()} 条价格逻辑错误")
        return False

    # 检查时间间隔
    time_diffs = df['timestamps'].diff().dropna()
    print(f"时间间隔: {time_diffs.min()} 到 {time_diffs.max()}")

    print("✅ 数据质量检查通过")
    return True

# 使用示例
validate_crypto_data(df)
```

---

## 3. 执行预测命令

### 3.1 基础预测示例

#### 单资产预测
```python
import pandas as pd
import sys
sys.path.append(".")
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. 加载模型和分词器
print("加载模型...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. 创建预测器
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. 准备数据
print("准备数据...")
df = pd.read_csv("btc_usdt_1h.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 设置预测参数
lookback = 400    # 使用过去400个时间点
pred_len = 120    # 预测未来120个时间点

# 准备输入数据
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. 执行预测
print("开始预测...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # 温度参数，控制随机性
    top_p=0.9,      # 核采样参数
    sample_count=1, # 样本数量
    verbose=True
)

# 5. 查看结果
print("预测结果:")
print(pred_df.head())
print(f"预测了 {len(pred_df)} 个时间点的价格")
```

#### 批量预测多个资产
```python
# 准备多个资产的数据
assets = ['btc_usdt_1h.csv', 'eth_usdt_1h.csv', 'bnb_usdt_1h.csv']
df_list = []
x_timestamp_list = []
y_timestamp_list = []

for asset_file in assets:
    df = pd.read_csv(asset_file)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

    df_list.append(x_df)
    x_timestamp_list.append(x_timestamp)
    y_timestamp_list.append(y_timestamp)

# 批量预测
print("开始批量预测...")
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 查看结果
for i, pred_df in enumerate(pred_df_list):
    print(f"资产 {i+1} 预测结果:")
    print(pred_df.head())
```

### 3.2 高级预测参数

#### 概率性预测
```python
# 生成多个预测样本以提高准确性
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.2,          # 较高温度，增加多样性
    top_p=0.95,     # 核采样，考虑更多可能性
    sample_count=5, # 生成5个样本并平均
    verbose=True
)
```

#### 保守预测
```python
# 更保守的预测设置
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.8,          # 较低温度，减少随机性
    top_p=0.8,      # 更严格的采样
    sample_count=3, # 较少样本
    verbose=True
)
```

### 3.3 预测结果分析

```python
import matplotlib.pyplot as plt

def analyze_prediction(historical_df, pred_df, asset_name="BTC/USDT"):
    """
    分析预测结果
    """
    # 计算预测准确性指标
    if len(historical_df) >= len(pred_df):
        actual_df = historical_df.iloc[-len(pred_df):]

        # 计算价格变化预测
        pred_change = (pred_df['close'].iloc[-1] - pred_df['close'].iloc[0]) / pred_df['close'].iloc[0] * 100
        actual_change = (actual_df['close'].iloc[-1] - actual_df['close'].iloc[0]) / actual_df['close'].iloc[0] * 100

        print(f"=== {asset_name} 预测分析 ===")
        print(f"预测价格变化: {pred_change:.2f}%")
        print(f"实际价格变化: {actual_change:.2f}%")
        print(f"预测误差: {abs(pred_change - actual_change):.2f}%")

        # 绘制对比图
        plt.figure(figsize=(12, 8))

        # 价格对比
        plt.subplot(2, 1, 1)
        plt.plot(historical_df['timestamps'], historical_df['close'], label='历史价格', color='blue')
        plt.plot(pred_df.index, pred_df['close'], label='预测价格', color='red', linestyle='--')
        plt.title(f'{asset_name} 价格预测对比')
        plt.legend()
        plt.grid(True)

        # 成交量对比
        plt.subplot(2, 1, 2)
        plt.plot(historical_df['timestamps'], historical_df['volume'], label='历史成交量', color='blue')
        plt.plot(pred_df.index, pred_df['volume'], label='预测成交量', color='red', linestyle='--')
        plt.title(f'{asset_name} 成交量预测对比')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# 使用示例
analyze_prediction(df, pred_df, "BTC/USDT")
```

---

## 4. Web UI 使用

### 4.1 启动 Web UI
```bash
cd webui
python run.py
```

### 4.2 使用步骤

1. **加载数据**
   - 点击"选择数据文件"
   - 选择你的加密货币数据文件（CSV 或 Feather 格式）
   - 系统会自动验证数据格式

2. **加载模型**
   - 选择模型大小（mini/small/base）
   - 选择计算设备（CPU/CUDA/MPS）
   - 点击"加载模型"

3. **设置预测参数**
   - **Lookback**: 历史数据长度（默认400）
   - **Prediction Length**: 预测长度（默认120）
   - **Temperature**: 预测随机性（0.1-2.0）
   - **Top-p**: 核采样参数（0.1-1.0）
   - **Sample Count**: 样本数量（1-5）

4. **选择时间窗口**
   - 使用滑块选择预测的时间范围
   - 系统会显示400个历史点+120个预测点

5. **开始预测**
   - 点击"开始预测"按钮
   - 等待预测完成

6. **查看结果**
   - 查看K线图对比
   - 查看预测数据表格
   - 下载预测结果

### 4.3 Web UI 高级功能

#### API 接口使用
```python
import requests
import json

# 加载数据
data = {
    "file_path": "/path/to/your/crypto_data.csv"
}
response = requests.post("http://localhost:7070/api/load-data", json=data)

# 加载模型
model_data = {
    "model_key": "kronos-small",
    "device": "cuda:0"
}
response = requests.post("http://localhost:7070/api/load-model", json=model_data)

# 执行预测
prediction_data = {
    "file_path": "/path/to/your/crypto_data.csv",
    "lookback": 400,
    "pred_len": 120,
    "temperature": 1.0,
    "top_p": 0.9,
    "sample_count": 1
}
response = requests.post("http://localhost:7070/api/predict", json=prediction_data)
result = response.json()
```

---

## 5. 高级功能

### 5.1 模型微调

#### 准备微调数据
```python
# 创建微调配置
from finetune.config import Config

config = Config()
config.qlib_data_path = "path/to/your/crypto/data"
config.instrument = "crypto"  # 自定义
config.dataset_begin_time = "2020-01-01"
config.dataset_end_time = "2024-12-31"
config.lookback_window = 90
config.predict_window = 10
```

#### 执行微调
```bash
# 1. 数据预处理
python finetune/qlib_data_preprocess.py

# 2. 微调分词器
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py

# 3. 微调预测器
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py

# 4. 回测评估
python finetune/qlib_test.py --device cuda:0
```

### 5.2 量化交易策略

#### 简单策略示例
```python
def simple_trading_strategy(pred_df, current_price, threshold=0.02):
    """
    基于预测的简单交易策略
    """
    # 计算预测价格变化
    pred_price = pred_df['close'].iloc[-1]
    price_change = (pred_price - current_price) / current_price

    if price_change > threshold:
        return "BUY", price_change
    elif price_change < -threshold:
        return "SELL", price_change
    else:
        return "HOLD", price_change

# 使用示例
signal, change = simple_trading_strategy(pred_df, current_btc_price)
print(f"交易信号: {signal}, 预期变化: {change:.2%}")
```

#### 风险管理
```python
def risk_management(pred_df, portfolio_value, max_risk=0.05):
    """
    风险管理函数
    """
    # 计算预测波动率
    pred_volatility = pred_df['close'].std() / pred_df['close'].mean()

    # 计算建议仓位
    max_position = max_risk / pred_volatility
    suggested_position = min(max_position, 1.0)  # 最大100%仓位

    return suggested_position, pred_volatility

# 使用示例
position, volatility = risk_management(pred_df, 10000, max_risk=0.05)
print(f"建议仓位: {position:.2%}, 预测波动率: {volatility:.2%}")
```

---

## 6. 常见问题

### 6.1 安装问题

**Q: 安装依赖时出现错误**
```bash
# 解决方案：升级 pip 并重新安装
pip install --upgrade pip
pip install -r requirements.txt
```

**Q: CUDA 相关错误**
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6.2 数据问题

**Q: 数据格式错误**
- 确保列名正确：`timestamps`, `open`, `high`, `low`, `close`, `volume`
- 检查时间戳格式是否正确
- 确保价格数据为数值类型

**Q: 数据量不足**
- Kronos 需要至少 400 个历史数据点
- 建议使用 1000+ 数据点以获得更好效果

### 6.3 预测问题

**Q: 预测结果不准确**
- 尝试调整温度参数（T）
- 增加样本数量（sample_count）
- 确保数据质量良好
- 考虑对特定市场进行微调

**Q: 内存不足**
- 减少 batch_size
- 使用较小的模型（Kronos-mini）
- 减少预测长度（pred_len）

### 6.4 性能优化

**Q: 预测速度慢**
- 使用 GPU 加速（CUDA/MPS）
- 使用批量预测（predict_batch）
- 选择较小的模型

**Q: 模型加载慢**
- 首次加载会下载模型，请耐心等待
- 可以预先下载模型到本地

---

## 📞 技术支持

如果遇到问题，请检查：
1. 项目 GitHub Issues
2. 控制台错误信息
3. 数据格式是否正确
4. 依赖是否完整安装

---

## 📄 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](./LICENSE) 文件。

---

*最后更新：2025年1月*
