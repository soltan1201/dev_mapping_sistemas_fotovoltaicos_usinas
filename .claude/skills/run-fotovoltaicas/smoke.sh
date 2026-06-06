#!/usr/bin/env bash
# smoke.sh — lança os apps e tira screenshots via Playwright
# Uso: bash .claude/skills/run-fotovoltaicas/smoke.sh [--port-dash 8051] [--port-st 8052]
# Saída: /tmp/dash_*.png  /tmp/streamlit_*.png

set -euo pipefail

PORT_DASH=${1:-8051}
PORT_ST=${2:-8052}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

INDEX="$ROOT/src/bd_images/predict_index_test.json"
if [ ! -f "$INDEX" ]; then
    echo "ERRO: $INDEX não encontrado. Crie dados de teste primeiro (ver SKILL.md)."
    exit 1
fi

# ── dashboard Dash ────────────────────────────────────────────────────────────
echo "[1/4] Iniciando dashboard Dash na porta $PORT_DASH..."
python3 "$ROOT/src/dash/dashboard_predict.py" \
    --index-file "$INDEX" --port "$PORT_DASH" > /tmp/dash.log 2>&1 &
DASH_PID=$!

# ── viewer Streamlit ──────────────────────────────────────────────────────────
echo "[2/4] Iniciando viewer Streamlit na porta $PORT_ST..."
streamlit run "$ROOT/src/show_colect_data/viewer_patches_npy.py" \
    --server.port "$PORT_ST" --server.headless true > /tmp/streamlit.log 2>&1 &
ST_PID=$!

echo "[3/4] Aguardando inicialização (8s)..."
sleep 8

# ── screenshots ───────────────────────────────────────────────────────────────
echo "[4/4] Tirando screenshots..."
python3 - << PYEOF
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto("http://localhost:${PORT_DASH}/", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/dash_smoke.png")
    print("  dashboard : /tmp/dash_smoke.png")

    page2 = browser.new_page(viewport={"width": 1400, "height": 900})
    page2.goto("http://localhost:${PORT_ST}/", wait_until="networkidle", timeout=15000)
    page2.wait_for_timeout(3000)
    page2.screenshot(path="/tmp/streamlit_smoke.png")
    print("  streamlit : /tmp/streamlit_smoke.png")

    browser.close()
PYEOF

kill $DASH_PID $ST_PID 2>/dev/null || true
echo "Concluído. Screenshots em /tmp/dash_smoke.png e /tmp/streamlit_smoke.png"
