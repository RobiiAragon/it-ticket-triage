#!/data/data/com.termux/files/usr/bin/bash
# IT ticket triage on Android via Termux (target: Galaxy Tab S9 Ultra, Snapdragon 8 Gen 2).
# Inference only: pure-Python tokenizer + ONNX Runtime (CPU / XNNPACK / NNAPI).
#
#   pkg install -y git && git clone https://github.com/RobiiAragon/it-ticket-triage
#   cd it-ticket-triage && bash scripts/android_termux_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "${PREFIX:-}" == *com.termux* ]] || { echo "This script is meant for Termux."; exit 1; }
VARIANTS="${VARIANTS:-int8}"   # e.g. VARIANTS=int8,fp32 (fp32 is a 87 MB download)

pkg update -y >/dev/null
pkg install -y python python-numpy python-onnxruntime git >/dev/null
python -m pip install -q --no-deps -e .
triage providers
for v in ${VARIANTS//,/ }; do triage download-models --variant "$v"; done

triage classify --variant "${VARIANTS%%,*}" \
  "Whole packing line 3 can't print shipping labels and the truck leaves at noon"

TAG=$(getprop ro.product.model | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
OUT="results/android-$TAG.json"
echo "==> evaluate $VARIANTS on CPU / XNNPACK / NNAPI -> $OUT"
triage evaluate --variants "$VARIANTS" --providers cpu,xnnpack,nnapi --out "$OUT"
echo "done. Share $OUT (termux-setup-storage && cp $OUT ~/storage/downloads/)"
