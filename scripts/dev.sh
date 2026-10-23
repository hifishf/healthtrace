#!/usr/bin/env bash
# HealthTrace 本地开发一键启动：Docker 基础设施 + 后端 + 前端
set -euo pipefail

cd "$(dirname "$0")/.."

# 0) 前置检查：venv 与 LLM key
if [[ ! -x .venv/bin/uvicorn ]]; then
    echo "❌ 未找到 .venv/bin/uvicorn，请先执行 uv sync" >&2
    exit 1
fi
if ! grep -q '^LLM_API_KEY=sk-' .env 2>/dev/null; then
    echo "⚠️  未检测到 .env 里的 LLM_API_KEY，聊天会失败（表单记录不受影响）" >&2
fi

# 1) 基础设施（幂等，已运行的会跳过）
echo "▶ [1/4] Docker 基础设施..."
docker compose up -d

# 2) 数据库迁移：Alembic 接管 schema（幂等，upgrade head 可重复执行）
echo "▶ [2/4] 数据库迁移（alembic upgrade head）..."
.venv/bin/alembic upgrade head

# 3) 后端：DeepSeek 是境内服务，直连更快更稳，故绕过可能失效的本地代理
echo "▶ [3/4] 后端 http://127.0.0.1:8000"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    .venv/bin/uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# 4) 前端
echo "▶ [4/4] 前端 http://localhost:8501"
.venv/bin/streamlit run frontend/streamlit_app.py &
FRONTEND_PID=$!

trap 'echo; echo "停止前后端..."; kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true' INT TERM

echo
echo "✅ 后端: http://127.0.0.1:8000  (pid $BACKEND_PID)"
echo "✅ 前端: http://localhost:8501   (pid $FRONTEND_PID)"
echo "   Ctrl+C 同时停止前后端"
wait
