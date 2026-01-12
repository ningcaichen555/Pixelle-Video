#!/bin/bash

# 检查上游是否有新更新
# 使用方法: ./check-updates.sh

echo "🔍 检查上游仓库更新..."

# 获取上游最新信息
git fetch upstream --quiet

# 比较本地main和上游main
LOCAL=$(git rev-parse main)
UPSTREAM=$(git rev-parse upstream/main)

if [ "$LOCAL" = "$UPSTREAM" ]; then
    echo "✅ 已是最新版本"
else
    echo "🆕 发现新更新！"
    echo ""
    echo "📋 新提交："
    git log --oneline main..upstream/main
    echo ""
    echo "💡 运行以下命令同步更新："
    echo "   ./sync-upstream.sh"
fi