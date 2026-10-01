# Ultradebate

Claude Code와 Codex CLI가 작업을 나누고 서로 검토할 수 있도록 만든 MCP 서버입니다. 실행 중인 CLI에 메시지를 전달하고, JSON inbox에 기록을 남기며, 공유 계획과 작업 상태를 관리합니다.

주로 사용한 환경은 Windows의 WSL2입니다. Claude는 WSL 터미널에서 실행하고, Codex는 같은 WSL 안의 tmux에서 실행합니다. 이 CLI 협업 경로에는 별도의 API 키가 필요하지 않습니다. 각 CLI에는 먼저 로그인되어 있어야 합니다.

## 설치와 MCP 등록

Python 3.10 이상, Bash, tmux와 로그인된 Claude Code·Codex CLI가 필요합니다. pane 탐색과 프로세스 확인에 Linux의 `/proc`를 사용하므로 이 문서의 운영 절차는 Linux/WSL 기준입니다. Windows PowerShell과 WSL의 실행 파일·설정을 혼용하지 마세요.

```bash
git clone https://github.com/jang2296/ultradebate.git
cd ultradebate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

command -v claude codex tmux
claude --version
codex --version

ULTRADEBATE_PYTHON="$(pwd -P)/.venv/bin/python"
claude mcp add --scope user --transport stdio ultradebate -- \
  "$ULTRADEBATE_PYTHON" -m ultradebate.server
codex mcp add ultradebate -- \
  "$ULTRADEBATE_PYTHON" -m ultradebate.server

claude mcp get ultradebate
codex mcp get ultradebate
```

현재 배포 경로는 이 Git 저장소입니다. `uvx ultradebate-mcp`가 PyPI에 게시된 패키지를 설치한다고 가정하지 마세요. 가상환경 Python을 **절대 경로**로 등록하므로 어느 프로젝트에서 CLI를 열어도 모듈을 찾을 수 있습니다.

Claude는 user 범위, Codex는 사용자 설정에 등록합니다. 이미 같은 이름의 서버가 있으면 기존 실행 경로부터 확인하세요. 설정 예시는 [Claude JSON](examples/claude_mcp.json), [Codex TOML](examples/codex_config.toml)에 있습니다.

Claude의 사용자 MCP 설정은 `~/.claude.json`, Codex는 `~/.codex/config.toml`입니다. Claude의 `~/.claude/settings.json`에 `mcpServers`를 붙여 넣지 않습니다. 변경 뒤 CLI 세션을 다시 열고 `/mcp`에서 연결 상태와 도구 목록을 확인하세요. 등록 방식은 [Claude Code MCP 문서](https://code.claude.com/docs/en/mcp)와 [Codex MCP 문서](https://developers.openai.com/codex/mcp)를 참고했습니다.

## 두 CLI 실행

두 CLI를 **같은 프로젝트 경로**에서 실행합니다. `/path/to/project`는 실제 작업할 프로젝트의 절대 경로로 바꿉니다.

첫 번째 WSL 터미널:

```bash
cd /path/to/project
claude
```

두 번째 WSL 터미널에서는 tmux를 먼저 열고 Codex를 실행합니다.

```bash
cd /path/to/project
tmux new-session -s debate-codex
codex
```

tmux에서 `Ctrl-b`, `d`로 분리해도 Codex 프로세스는 계속 실행됩니다. 다시 연결할 때는 `tmux attach -t debate-codex`를 사용합니다. 세션 이름은 고정하지 않아도 됩니다. pane을 확인하는 명령은 다음과 같습니다.

```bash
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_id} #{pane_current_command} #{pane_current_path}'
```

각 CLI에서 MCP 도구로 자신을 등록합니다. 아래는 도구 호출 예시이며 셸 명령이 아닙니다.

```text
ultradebate_v2_session_attach(
  agent="claude", cwd="/path/to/project", auto_detect_from_cwd=true
)
ultradebate_v2_session_attach(
  agent="codex", cwd="/path/to/project", auto_detect_from_cwd=true
)
ultradebate_v2_session_status(cwd="/path/to/project")
```

두 쪽이 등록되면 `binding_status="dual_active"`인지 확인합니다. 자동 탐색에 실패하면 실제 CLI PID와 tmux 정보를 명시합니다. pane을 다시 열거나 CLI를 재시작한 뒤에는 다시 등록하세요. 동일 프로젝트에서 Codex를 여러 개 실행한 경우에는 전송 도구의 `pane` 인자로 대상을 지정하세요.

## 메시지 전달과 실제 Enter

Claude에서 실행 중인 Codex에 보냅니다. pane 이름은 실제 값으로 바꿉니다.

```text
ultradebate_v2_send_to_codex(
  sender="claude", text="Review the changes and report any blocking issues.",
  pane="debate-codex:0.0"
)
```

Codex에서 tmux 밖의 WSL 터미널에 있는 Claude에 보낼 때는 JSON inbox와 다음 절의 알림 경로를 사용합니다.

```text
ultradebate_v2_send_to_claude(
  sender="codex", text="Review complete. Please read the findings.",
  tmux_inject=false
)
```

메시지 끝에 문자열 `\n`을 붙이거나 입력 칸에 줄바꿈만 넣는 것으로 제출되지 않습니다. **Claude가 Codex에 보낼 때도 사용자가 Enter를 누르는 것과 같은 별도 키 입력이 필요합니다.** Claude 자체를 tmux에서 실행하는 경우에도 같은 원칙입니다.

서버의 전송 순서는 다음과 같습니다.

1. 대상 JSONL inbox에 메시지를 기록합니다.
2. 본문의 줄바꿈·탭·제어 문자를 정리하고 `tmux load-buffer`와 `paste-buffer -p`로 하나의 텍스트로 붙여 넣습니다.
3. 붙여 넣기가 반영될 시간을 둔 뒤 별도의 제출 키를 보냅니다. Codex에는 `C-m` 하나, Claude에는 `Enter`를 사용합니다.
4. pane 출력으로 제출 상태를 확인합니다.

여러 줄짜리 본문과 Enter를 한 번의 `tmux send-keys`에 섞어 보내지 마세요. 본문이 중간에 제출되거나 입력 칸에 줄바꿈만 남았던 문제의 원인이었습니다. 전송은 위 MCP 도구로 처리하세요. Codex 경로에는 Tab이나 Escape+Enter를 쓰지 않습니다.

`ok=true`는 inbox 기록 성공을 포함하는 결과입니다. 상대가 읽었다는 뜻까지는 아닙니다. `tmux_state`와 후속 응답을 함께 확인하세요.

| `tmux_state` | 의미 |
| --- | --- |
| `submitted` | 작업 표시를 관찰했거나 전송 경로가 완료됨. 실제 수신은 응답이나 ACK로 확인 |
| `submit_unverified` | 붙여 넣기·키 입력은 수행했으나 제출 확인이 불충분함. 중복 전송 전에 inbox와 응답 확인 |
| `no_pane` / `paste_failed` | pane 탐색 또는 전송 실패. JSON 메시지는 보존됨 |
| `skipped` | `tmux_inject=false`로 inbox에만 전송 |
| `queued_busy` / `queued_via_tab` | Codex 이외 대상의 기존 busy 처리 결과. Codex에는 Tab 큐를 쓰지 않음 |

## Claude의 JSON 알림은 항상 켜두기

Codex가 inbox에 기록하는 것과 Claude가 즉시 알림을 받는 것은 별개입니다. **협업 중에는 Claude의 수신 감시를 계속 유지하세요.**

`scripts/claude_inbox_watcher.sh`는 `/tmp/ultradebate_v2/inbox/claude.jsonl`의 새 행을 감시해 JSON 이벤트를 한 줄씩 출력합니다. 발신자·우선순위·종류·`msg_id`를 알리고, 본문은 MCP로 읽습니다.

사용 중인 Claude 호스트에 `Monitor` 도구가 제공되는 경우, Claude에게 다음 감시를 **persistent 구독으로 시작하고 유지**하도록 요청합니다. 설치한 저장소의 절대 경로로 바꾸세요.

```text
Monitor(
  command="bash /path/to/ultradebate/scripts/claude_inbox_watcher.sh claude",
  persistent=true
)
```

`Monitor`는 이 MCP가 제공하는 도구가 아닙니다. 모든 Claude Code 버전에 있다고 가정하지 마세요. `nohup`으로 로그 파일에만 출력하는 워처는 Claude 대화를 깨우지 않습니다. 호스트가 stdout 이벤트를 대화 알림으로 전달하는 구독까지 활성화되어 있어야 합니다.

`Monitor`가 없는 호스트에서는 기존 메시지를 먼저 읽고 다음 메시지를 기다리는 MCP 경로를 사용합니다.

```text
ultradebate_v2_priority_inbox_read(agent="claude", drain=true)
ultradebate_v2_wait_for_peer_message(
  agent="claude", timeout_seconds=30, min_priority="low"
)
```

반환되면 메시지를 처리하고 읽기와 대기를 반복합니다. 이 경로는 호출이 활성화된 동안만 기다리며, CLI가 완전히 idle 상태로 돌아간 후 자동으로 깨우지는 않습니다. 협업 중에는 대기를 다시 걸어두세요. MCP 클라이언트의 도구 제한 시간보다 짧게 기다립니다.

워처와 대기는 시작 이후 도착한 메시지를 대상으로 합니다. 시작·재시작 직후에는 `priority_inbox_read`로 미처리 메시지를 먼저 확인하세요. 알림의 `msg_id`로 본문을 식별하고, 같은 메시지를 중복 작업으로 실행하지 않도록 drain과 수신 응답을 기록합니다.

터미널에서 알림 출력 자체를 확인할 수도 있습니다. 다음 프로세스는 종료할 때까지 실행됩니다.

```bash
bash scripts/claude_inbox_watcher.sh claude
```

이 상태에서 Codex가 `send_to_claude`를 호출했을 때 `event="V2_INBOX_NEW"`인 JSON 한 줄이 출력되는지 확인합니다. 이어서 Claude의 persistent 구독에서 같은 알림이 실제 대화로 전달되는지 확인하세요.

## 기본 협업 순서와 주요 도구

1. 두 CLI를 등록하고 `session_status`를 확인합니다.
2. Claude 수신 감시를 켜고 각자의 미처리 inbox를 읽습니다.
3. 파일 범위를 나누고 완료 조건을 포함해 작업을 전달합니다.
4. 상대가 수신 ACK나 질문을 보내고 작업합니다.
5. 결과를 공유해 서로 검토한 뒤 테스트로 확인합니다.

긴 셸 작업은 `ultradebate_v2_run_long_job`으로 시작하고 반환된 `job_id`를 `ultradebate_v2_check_job`으로 조회합니다. 관찰 시간이 끝났다는 이유로 같은 작업을 새로 실행하지 말고 기존 job 상태를 먼저 확인하세요.

| 용도 | 도구 |
| --- | --- |
| 환경 확인 | `ultradebate_check_environment` |
| CLI 등록·상태·복구 | `ultradebate_v2_session_attach`, `ultradebate_v2_session_status`, `ultradebate_v2_session_resume` |
| 전송 | `ultradebate_v2_send_to_peer`, `ultradebate_v2_send_to_codex`, `ultradebate_v2_send_to_claude` |
| JSON 기록·수신 | `ultradebate_v2_priority_send`, `ultradebate_v2_priority_inbox_read`, `ultradebate_v2_wait_for_peer_message` |
| 작업·검토 | `ultradebate_v2_task_create`, `ultradebate_v2_task_implement`, `ultradebate_v2_task_review`, `ultradebate_v2_task_status` |
| 백그라운드 작업 | `ultradebate_v2_run_long_job`, `ultradebate_v2_check_job`, `ultradebate_v2_list_jobs`, `ultradebate_v2_cancel_job` |
| 미답변 질문 | `ultradebate_v2_open_questions_register`, `ultradebate_v2_open_questions_respond`, `ultradebate_v2_open_questions_pending` |
| 배분 추천 | `ultradebate_v2_predict_allocation`, `ultradebate_v2_allocator_info`, `ultradebate_v2_allocator_reload` |

전체 인자와 도구 목록은 MCP 클라이언트에서 확인합니다. 낮은 단계의 `priority_send`는 Codex 대상 긴급 메시지에서 tmux 깨우기도 시도합니다. JSON 기록만 필요하면 `auto_wake=false`를 지정하세요. 일반 전송에는 `send_to_peer` 또는 전용 wrapper를 사용합니다.

## 운영 중 겪은 문제

| 증상 | 확인과 해결 |
| --- | --- |
| Codex에 본문은 보이는데 실행이 안 됨 | 본문 줄바꿈과 제출 키를 구분. 안전한 paste-buffer 경로와 별도 `C-m` 사용 |
| Codex의 응답이 Claude에 묻힘 | Claude의 persistent 알림 구독이 살아 있는지 확인. 워처 로그만으로는 부족함 |
| 워처에 아무것도 안 찍힘 | 최신 경로는 `/tmp/ultradebate_v2/inbox/<agent>.jsonl`. 기존 `<agent>/inbox/*.json`과 다름 |
| 재시작 뒤 예전 메시지를 놓침 | 워처는 과거 행을 재알림하지 않음. `priority_inbox_read`로 먼저 확인 |
| 다른 Codex pane으로 전달됨 | 두 CLI의 CWD·등록·pane 확인. 복수 pane은 명시적으로 지정 |
| 새 도구가 안 보임 | 기존 MCP 프로세스가 이전 모듈을 로드했을 수 있음. 설정 확인 후 연결 재시작 |
| JSON 인코딩 오류나 HTTP 400 | tmux 캡처의 잘못된 UTF-8·단독 surrogate를 처리하는 `text_safety` 경로 사용 |
| 긴 작업이 끝났는지 불명확함 | 기존 `job_id`의 상태와 로그를 확인한 뒤 후속 작업 결정 |

`codex_inbox_watcher.sh`, `codex_pane_autodetect.sh`, `codex_auto_pull_daemon.sh`는 이전 파일 inbox 방식의 보조 스크립트입니다. 최신 JSONL 경로의 범용 알림 설정으로 쓰지 마세요. 기본 협업은 MCP 전송 도구와 Claude 수신 감시로 구성합니다.

## 데이터와 접근 범위

메시지·세션 등록·공유 계획은 `/tmp/ultradebate_v2/` 아래에 저장됩니다. 이전 라운드 토론 도구는 `~/.ultradebate/`를 사용합니다. `/tmp` 데이터는 재부팅이나 정리 정책으로 삭제될 수 있으므로 필요한 결과는 따로 저장하세요.

현재 v2 inbox는 프로젝트별 분리 큐가 아니라 로컬의 공용 agent 큐입니다. 프로젝트 등록과 pane 탐색이 있어도 여러 프로젝트의 동시 운영을 완전히 격리하지는 않습니다. 기본 사용 범위는 같은 WSL 사용자와 하나의 협업 프로젝트입니다.

MCP는 셸 작업과 tmux 입력을 수행합니다. 파일·메시지에 비밀키나 개인 자료를 넣지 마세요. CLI의 기존 승인·권한 설정을 유지하며 설치를 위해 권한 확인을 해제할 필요는 없습니다. 별도 CLI worker에 API 키를 사용할 때도 환경변수로 전달하고 커밋하지 않습니다.

공개본에는 소스·테스트·범용 운영 및 학습 스크립트를 포함합니다. 실제 대화·사용량 로그, 학습 데이터·직렬화 모델, 개인 프로젝트 계획, 발표자 신상 자료와 로컬 설정은 제외했습니다. `.gitignore`에서 실행 결과와 로그도 제외합니다.

## 선택 사항: 작업 배분 모델과 이전 라운드 도구

CLI 메시지 전달에 MLP 모델은 필요하지 않습니다. 추천을 사용하려면 `python -m pip install -e '.[mlp]'`로 선택 의존성을 설치하고 모델을 직접 학습합니다. 개인 로그 없이 합성 데이터로 실행하는 절차는 [presentation/README.md](presentation/README.md)에 있습니다.

모델·메타데이터 경로는 `ULTRADEBATE_ALLOCATOR_MODEL`, `ULTRADEBATE_ALLOCATOR_META`로 지정합니다. 로더는 기본적으로 `model_sha256`과 파일의 sha256을 확인합니다. `ULTRADEBATE_ALLOCATOR_LEGACY=1`은 hash가 없는 레거시 파일의 검증을 해제하는 옵션이며 기본 사용에서는 켜지 않습니다. sha256 일치는 출처의 신뢰성을 증명하지 않습니다. joblib 모델은 pickle을 사용하므로 직접 학습했거나 신뢰하는 출처의 파일만 로드하세요.

이전 `ultradebate_start`·`round`·`synthesize` 등과 Gemini worker도 코드에 남아 있습니다. 이 방식은 별도 CLI subprocess를 실행할 수 있으므로 실행 중인 Claude–Codex 세션 협업과 구분합니다. 위 기본 운영 절차는 해당 도구를 호출하지 않습니다.

## 개발 검증

```bash
python -m pip install -e '.[dev,mlp]'
python -m pytest tests
python -m ruff check --select E9,F63,F7,F82 src tests scripts
python -m ultradebate.cli
```

마지막 명령은 기본 mock 검증으로 실제 CLI를 호출하지 않습니다. 실제 로그인·CLI 버전·호스트 알림 구독은 각 환경에서 별도로 확인해야 합니다. 협업 지침 예시는 [examples/skills/ultradebate/SKILL.md](examples/skills/ultradebate/SKILL.md)에 있습니다.

Claude에서 `/ultradebate`로 이 지침을 사용하려면 저장소 루트에서 다음을 실행하고 Claude 세션을 다시 엽니다. 기존에 같은 이름의 스킬이 있으면 먼저 내용을 비교하세요. MCP 등록과 스킬 설치는 별개입니다.

```bash
mkdir -p "$HOME/.claude/skills/ultradebate"
cp examples/skills/ultradebate/SKILL.md "$HOME/.claude/skills/ultradebate/SKILL.md"
```

Claude 입력 예시:

```text
/ultradebate 현재 변경을 Codex와 검토해줘. Claude는 설계와 검토를 맡고,
Codex는 구현과 테스트를 맡아줘. 파일 수정 범위를 나누고 JSON 수신 감시는 유지해줘.
```

## 라이선스

[MIT](LICENSE).
