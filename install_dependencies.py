#!/usr/bin/env python3.11
"""
Kronos项目依赖安装脚本
适用于Python 3.11
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """执行命令并显示结果"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 11:
        print("❌ 需要Python 3.11或更高版本")
        return False

    print("✅ Python版本符合要求")
    return True

def install_dependencies():
    """安装依赖"""
    print("=" * 60)
    print("🚀 开始安装Kronos项目依赖")
    print("=" * 60)

    # 检查Python版本
    if not check_python_version():
        return False

    # 升级pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "升级pip"):
        return False

    # 安装核心依赖
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "安装核心依赖"):
        return False

    # 安装Web UI依赖
    if os.path.exists("webui/requirements.txt"):
        if not run_command(f"{sys.executable} -m pip install -r webui/requirements.txt", "安装Web UI依赖"):
            return False

    # 验证安装
    print("\n🔍 验证依赖安装...")
    try:
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        from huggingface_hub import hf_hub_download
        print("✅ 核心依赖验证成功")
        print(f"   NumPy: {np.__version__}")
        print(f"   Pandas: {pd.__version__}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")

        # 测试Web UI依赖
        import flask
        import plotly
        print("✅ Web UI依赖验证成功")
        print(f"   Flask: {flask.__version__}")
        print(f"   Plotly: {plotly.__version__}")

    except ImportError as e:
        print(f"❌ 依赖验证失败: {e}")
        return False

    # 测试Kronos模型库
    print("\n🤖 测试Kronos模型库...")
    try:
        sys.path.append('.')
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos模型库导入成功")
    except ImportError as e:
        print(f"⚠️  Kronos模型库导入失败: {e}")
        print("   这可能是正常的，模型会在首次使用时下载")

    print("\n" + "=" * 60)
    print("🎉 依赖安装完成！")
    print("=" * 60)
    print("\n📋 下一步操作:")
    print("1. 启动Web UI: cd webui && python app.py")
    print("2. 运行预测示例: python examples/prediction_example.py")
    print("3. 查看使用指南: 打开 Kronos_加密货币预测使用指南.md")

    return True

def main():
    """主函数"""
    try:
        success = install_dependencies()
        if success:
            print("\n✅ 安装成功完成！")
            sys.exit(0)
        else:
            print("\n❌ 安装过程中遇到错误")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 安装过程中发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

