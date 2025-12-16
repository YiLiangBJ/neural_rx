#!/bin/bash
# Neural RX 项目环境配置脚本

echo "🔧 加载环境配置..."

# 加载 .env 文件
if [ -f .env ]; then
    source .env
    echo "✅ 代理配置已加载"
    echo "   HTTP_PROXY: $HTTP_PROXY"
    echo "   HTTPS_PROXY: $HTTPS_PROXY"
else
    echo "⚠️  .env 文件不存在，使用 .env.example 作为模板创建一个"
    exit 1
fi

echo ""
echo "🐍 创建虚拟环境..."
uv venv --python python

if [ $? -eq 0 ]; then
    echo "✅ 虚拟环境创建成功"
    echo ""
    echo "📦 激活虚拟环境并安装依赖:"
    echo "   source .venv/bin/activate"
    echo "   uv pip install -r requirements.txt"
else
    echo "❌ 虚拟环境创建失败"
    exit 1
fi
