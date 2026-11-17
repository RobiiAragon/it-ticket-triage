#!/usr/bin/env bash
# macOS end-to-end check: lint, tests, (optional) retrain, evaluation on CPU + CoreML,
# CLI and dashboard smoke tests.
#   scripts/mac_check.sh             # download released models, evaluate
#   TRAIN=1 scripts/mac_check.sh     # re-split, fine-tune (MPS), export, quantize, evaluate
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[[ -n "$PY" ]] || { echo "Python 3.11+ not found (brew install python@3.12)"; exit 1; }
[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null
python -m pip install -q --upgrade pip
EXTRA="dev"; [[ -n "${TRAIN:-}" ]] && EXTRA="dev,train"
python -m pip install -q -e ".[$EXTRA]"

echo "==> lint + tests"
ruff check . && ruff format --check . && pytest -q -W ignore::DeprecationWarning
triage providers

if [[ -n "${TRAIN:-}" ]]; then
  echo "==> prepare + fine-tune + export + quantize"
  triage prepare
  triage train
else
  triage download-models --variant all
fi

echo "==> evaluate (all variants x CPU/CoreML, plus TF-IDF baseline)"
triage evaluate --providers cpu,coreml,qnn,cuda --baseline --out results/local-mac.json

echo "==> CLI + API smoke test"
triage classify "Whole packing line 3 can't print shipping labels and the truck leaves at noon"
triage serve --metrics results/local-mac.json --port 8768 > /dev/null 2>&1 &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -sf http://127.0.0.1:8768/health >/dev/null && break; sleep 0.5; done
curl -sf -X POST http://127.0.0.1:8768/classify -H 'Content-Type: application/json' \
  -d '{"text":"I clicked a link in an email and typed my password, now I get MFA prompts"}'
echo
echo "OK - triage serve, then open http://127.0.0.1:8000"
