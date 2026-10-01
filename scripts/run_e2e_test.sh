#!/usr/bin/env bash
# Codex CLI 설치 직후 전체 시스템 검증을 위한 단축 스크립트.
#
# 동작:
#   1. 모든 CLI 와 API 키 존재 확인
#   2. mock 모드 테스트 통과 확인
#   3. live 모드 (실제 3-AI 토론) 테스트 — 비용 ~$0.20-0.50

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${ULTRADEBATE_PYTHON:-$(command -v python3)}"
if [ ! -x "$PYTHON_BIN" ] || ! "$PYTHON_BIN" -c 'import mcp, pytest' >/dev/null 2>&1; then
  echo "오류: MCP SDK와 pytest가 설치된 Python이 필요합니다: $PYTHON_BIN" >&2
  echo '       가상환경을 활성화하거나 ULTRADEBATE_PYTHON을 지정하세요.' >&2
  exit 2
fi

echo "============================================"
echo "  ultradebate-mcp 통합 검증"
echo "============================================"
echo

echo "[1/3] 환경 진단"
echo "--------------------------------------------"
PYTHONPATH=src "$PYTHON_BIN" -c "
import asyncio, json
from ultradebate import server
print(json.dumps(asyncio.run(server.ultradebate_check_environment()), indent=2, ensure_ascii=False))
"
echo

echo "[2/3] 단위/통합 테스트 (mock)"
echo "--------------------------------------------"
PYTHONPATH=src "$PYTHON_BIN" -m pytest tests/ -q
echo

echo "[3/3] LIVE E2E 테스트 (실제 3-AI 호출)"
echo "--------------------------------------------"
if ! command -v codex >/dev/null 2>&1; then
  echo "⚠️  codex 미설치 — bash scripts/install_codex.sh 먼저 실행하세요."
  exit 1
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ -z "${OPENAI_API_KEY:-}" ] || [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
  echo "⚠️  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY 중 일부 누락."
  echo "   .env 파일을 .env.example 참고하여 채우고 source 한 뒤 재실행하세요."
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" -m ultradebate.cli --live --rounds 2 --mode strict \
    --topic "$(cat <<'EOF'
파이썬 GIL 제거 (PEP 703) 에 찬성하는가?
- 멀티스레드 성능 향상의 실질적 가치
- 단일 스레드 성능 영향 (예상 -10%)
- C 확장 호환성 영향
- 라이브러리 생태계 마이그레이션 비용
세 AI 가 각자 다른 관점에서 토론하라.
EOF
)"
