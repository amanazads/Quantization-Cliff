#!/usr/bin/env bash
# Build a GENUINE BF16 GGUF and register it with Ollama.
#
# WHY THIS EXISTS
# ---------------
# The published Ollama tag `qwen2.5:3b-instruct-fp16` is IEEE half precision
# (F16), not bfloat16. F16 and BF16 both use 16 bits but split them differently:
# F16 has 10 mantissa / 5 exponent bits, BF16 has 7 mantissa / 8 exponent bits.
# Qwen2.5 was trained in BF16, so serving it as F16 is itself a format conversion.
# Since BF16 is the REFERENCE arm that every delta is measured against, that
# substitution is recorded as deviation DEV-BF16-OLLAMA-1 and is worth removing.
#
# CAVEAT YOU MUST CHECK
# ---------------------
# llama.cpp's Metal backend has only partial BF16 support and may upcast or fall
# back to CPU. If it does, the arm measures the BF16 STORAGE format but not BF16
# arithmetic, and the run is slower. Verify before trusting it:
#
#   ollama run qwen2.5-3b-instruct-bf16 "hi" --verbose
#
# and check the server log for the compute backend actually selected. Record what
# you observe in the findings report rather than assuming.
#
# REQUIREMENTS: git, python3, ~15 GB free disk, network access to Hugging Face.

set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct}"
WORK="${WORK:-$HOME/.cache/ps5-bf16}"
TAG="${TAG:-qwen2.5-3b-instruct-bf16}"

mkdir -p "$WORK"
cd "$WORK"

echo "== 1. fetch llama.cpp conversion tooling =="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
python3 -m pip install --quiet -r llama.cpp/requirements.txt

echo "== 2. download the original safetensors checkpoint =="
python3 -m pip install --quiet "huggingface_hub[cli]"
huggingface-cli download "$MODEL_REPO" --local-dir ./hf-model

echo "== 3. convert to BF16 GGUF (NOT f16) =="
python3 llama.cpp/convert_hf_to_gguf.py ./hf-model \
  --outtype bf16 \
  --outfile "./${TAG}.gguf"

echo "== 4. register with Ollama =="
cat > Modelfile <<EOF
FROM ./${TAG}.gguf
# Decoding parameters are supplied per request by the runner; nothing is pinned
# here, so this Modelfile cannot silently differ from the other arms.
EOF
ollama create "$TAG" -f Modelfile

echo
echo "Created Ollama model: $TAG"
echo
echo "Verify it really is BF16:"
echo "  ollama show $TAG --modelfile | grep -i quant"
echo
echo "Then point the reference arm at it:"
echo "  edit configs/experiments/bf16.yaml -> backends.ollama.model_tag: \"$TAG\""
echo "  set quantization_format: \"GGUF BF16\""
echo "  and REMOVE deviation DEV-BF16-OLLAMA-1 only once you have confirmed the"
echo "  Metal backend is not silently upcasting."
echo
echo "Changing the reference arm invalidates comparisons against previously"
echo "collected arms. Re-run ALL FOUR precisions afterwards."
