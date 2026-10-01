#!/usr/bin/env bash
# Codex CLI 설치 부트스트랩 — 사용자가 OpenAI Codex CLI를 받은 직후 실행하기 위한 스크립트.
#
# 2026-04 기준 OpenAI Codex CLI 공식 설치 경로 후보:
#   1) npm:   npm install -g @openai/codex
#   2) Homebrew: brew install openai-codex
#   3) GitHub release: https://github.com/openai/codex/releases (수동 binary)
#
# 이 스크립트는 위 후보를 순서대로 시도하고, 모두 실패하면 사용자에게 수동
# 설치 경로를 안내합니다. 설치 후 `codex --version` 으로 검증합니다.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ULTRADEBATE_PYTHON:-$(command -v python3)}"

cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
warn()  { printf '\033[33m%s\033[0m\n' "$*"; }

cyan "=========================================="
cyan "  Codex CLI installer for ultradebate-mcp"
cyan "=========================================="
echo

if command -v codex >/dev/null 2>&1; then
  green "✓ codex 이미 설치됨: $(codex --version 2>/dev/null || echo '(--version 미지원)')"
  green "  → 추가 작업 불필요. 다음 명령으로 E2E 테스트:"
  echo "    cd $PROJECT_ROOT && PYTHONPATH=src \"$PYTHON_BIN\" -m ultradebate.cli --live --rounds 2"
  exit 0
fi

warn "codex 가 PATH 에 없습니다. 자동 설치를 순차 시도합니다..."
echo

# Attempt 1: npm
if command -v npm >/dev/null 2>&1; then
  cyan "[1/3] npm install -g @openai/codex 시도..."
  if npm install -g @openai/codex 2>&1 | tail -20; then
    if command -v codex >/dev/null 2>&1; then
      green "✓ npm 으로 설치 성공"
    fi
  else
    warn "  npm 설치 실패 (패키지 미공개 또는 권한 문제일 수 있음)"
  fi
fi

# Attempt 2: Homebrew (macOS only typically)
if ! command -v codex >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    cyan "[2/3] brew install openai-codex 시도..."
    brew install openai-codex 2>&1 | tail -20 || warn "  brew 설치 실패"
  fi
fi

# Final check
if command -v codex >/dev/null 2>&1; then
  echo
  green "✅ Codex CLI 설치 완료"
  green "   버전: $(codex --version 2>/dev/null || echo '미상')"
  echo
  cyan "다음 단계:"
  echo "  1) export OPENAI_API_KEY=sk-..."
  echo "  2) export ANTHROPIC_API_KEY=sk-ant-..."
  echo "  3) export GEMINI_API_KEY=AIza..."
  echo "  4) cd $PROJECT_ROOT && PYTHONPATH=src \"$PYTHON_BIN\" -m ultradebate.cli --live --rounds 2"
else
  echo
  red "✗ 자동 설치 실패."
  echo
  warn "수동 설치 안내:"
  echo "  • npm: https://www.npmjs.com/package/@openai/codex"
  echo "  • GitHub releases: https://github.com/openai/codex/releases"
  echo "  • OpenAI 공식 문서: https://developers.openai.com/codex"
  echo
  warn "설치 완료 후 codex 가 PATH 에 있는지 확인:"
  echo "  command -v codex && codex --version"
  exit 1
fi
