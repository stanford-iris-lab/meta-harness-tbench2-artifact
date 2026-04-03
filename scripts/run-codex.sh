#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${CODEX_ENV_FILE:-${ROOT_DIR}/.env.codex}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${CODEX_API_BASE:=https://codex-openai-wrapper/v1}"
: "${CODEX_API_KEY:=x}"
: "${HARBOR_MODEL:=openai/gpt-5.4}"
: "${HARBOR_DATASET:=hello-world@1.0}"
: "${HARBOR_TASK:=}"

export CODEX_API_BASE
export CODEX_API_KEY

cd "${ROOT_DIR}"

if [[ $# -eq 0 ]]; then
  cmd=(
    harbor run
    -d "${HARBOR_DATASET}"
    --agent-import-path agent:AgentHarness
    -m "${HARBOR_MODEL}"
    -n 1
  )
  if [[ -n "${HARBOR_TASK}" ]]; then
    cmd+=(-t "${HARBOR_TASK}")
  fi
  exec "${cmd[@]}"
fi

exec harbor run \
  --agent-import-path agent:AgentHarness \
  -m "${HARBOR_MODEL}" \
  "$@"
