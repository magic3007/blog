#!/usr/bin/env bash
set -euo pipefail

info()  { printf "\033[34m[INFO]\033[0m  %s\n" "$1"; }
ok()    { printf "\033[32m[OK]\033[0m    %s\n" "$1"; }
error() { printf "\033[31m[ERROR]\033[0m %s\n" "$1"; exit 1; }

cd "$(dirname "$0")"

# ── 检查前置依赖 ──────────────────────────────────────────
info "检查前置依赖..."

command -v ruby  >/dev/null 2>&1 || error "未找到 ruby，请先安装 Ruby (推荐 >= 3.1)"
command -v bundle >/dev/null 2>&1 || error "未找到 bundler，请运行: gem install bundler"
command -v node  >/dev/null 2>&1 || error "未找到 node，请先安装 Node.js (推荐 >= 18)"
command -v npm   >/dev/null 2>&1 || error "未找到 npm，请先安装 Node.js"

ok "ruby   $(ruby --version | awk '{print $2}')"
ok "bundle $(bundle --version | awk '{print $NF}')"
ok "node   $(node --version)"
ok "npm    $(npm --version)"

# ── 安装 Ruby 依赖 ────────────────────────────────────────
info "安装 Ruby gem 依赖..."
bundle install
ok "Ruby 依赖安装完成"

# ── 安装 Node.js 依赖 ─────────────────────────────────────
info "安装 Node.js 依赖..."
npm install
ok "Node.js 依赖安装完成"

# ── 验证构建 ──────────────────────────────────────────────
info "验证 Jekyll 构建..."
JEKYLL_ENV=production bundle exec jekyll build --quiet
ok "Jekyll 构建成功"

# ── 完成 ──────────────────────────────────────────────────
printf "\n\033[32m✔ 环境配置完成！\033[0m\n"
echo ""
echo "常用命令："
echo "  make              # 启动本地开发服务器 (livereload)"
echo "  npm run serve     # 启动开发服务器"
echo "  npm run build     # 生产环境构建"
echo ""
