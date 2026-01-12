#!/bin/bash

# Pixelle-Video 上游同步脚本
# 使用方法: ./sync-upstream.sh

set -e  # 遇到错误立即退出

echo "🔄 开始同步上游仓库更新..."

# 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo "❌ 检测到未提交的更改，请先提交或暂存："
    git status --porcelain
    exit 1
fi

# 获取当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 当前分支: $CURRENT_BRANCH"

# 检查上游远程是否存在
if ! git remote get-url upstream > /dev/null 2>&1; then
    echo "➕ 添加上游仓库..."
    git remote add upstream https://github.com/AIDC-AI/Pixelle-Video.git
fi

echo "📥 获取上游更新..."
git fetch upstream

echo "🔀 切换到主分支并合并上游更新..."
git checkout main
git merge upstream/main

echo "📤 推送更新到远程仓库..."
git push origin main

# 如果当前不在主分支，询问是否更新开发分支
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo ""
    read -p "🤔 是否将更新合并到开发分支 '$CURRENT_BRANCH'? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔀 切换回开发分支并合并更新..."
        git checkout "$CURRENT_BRANCH"
        
        # 尝试合并，如果有冲突会停止
        if git merge main; then
            echo "✅ 成功合并到 $CURRENT_BRANCH"
        else
            echo "⚠️  合并时发现冲突，请手动解决后运行："
            echo "   git add ."
            echo "   git commit -m 'Resolve merge conflicts'"
            exit 1
        fi
    else
        git checkout "$CURRENT_BRANCH"
        echo "ℹ️  已切换回 $CURRENT_BRANCH，如需合并请手动执行："
        echo "   git merge main"
    fi
fi

echo ""
echo "🎉 上游同步完成！"
echo "📊 最近的提交："
git log --oneline -5