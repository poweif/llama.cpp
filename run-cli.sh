#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

use_vulkan=0
use_qwen3=0
use_gemma_mtp=0
passthrough=()
for arg in "$@"; do
    case "$arg" in
        --vulkan)     use_vulkan=1 ;;
        --qwen3)      use_qwen3=1 ;;
        --gemma-mtp)  use_gemma_mtp=1 ;;
        *)            passthrough+=("$arg") ;;
    esac
done
set -- "${passthrough[@]+"${passthrough[@]}"}"

if (( use_vulkan )); then
    export HIP_VISIBLE_DEVICES=""
fi

if (( use_gemma_mtp )); then
    args=(
	-m /home/poweif/models/gemma-4-26B-A4B-it-Q8_0.gguf
	-md /home/poweif/models/gemma-4-26B-A4B-it-assistant-Q8_0.gguf
	--spec-type draft-mtp
	--spec-draft-n-max 3
	-ngl 99
	--spec-draft-ngl 99
	-fa on
	-c 262144
	-n 16384
	-b 4096
	-ub 512
	--jinja
	--reasoning off
	#--reasoning-budget 1024
	# NOTE: do NOT add --cache-type-k/v q8_0 here — quantized KV breaks frozen-KV attention
	--no-warmup
    )

elif (( use_qwen3 )); then
    args=(
	-m /home/poweif/models/Qwen_Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf
	-ngl 99
	-fa on
	-c 262144
	-n 16384
	-b 4096
	-ub 512
	--no-context-shift
	--jinja
	--cache-type-k q8_0
	--cache-type-v q8_0
	--no-warmup
    )

else
    args=(
	-m /home/poweif/models/gemma-4-26B-A4B-it-Q8_0.gguf
	-ngl 99
	-fa on
	-c 262144
	-n 16384
	-b 4096
	-ub 512
	--no-context-shift
	--jinja
	--cache-type-k q8_0
	--cache-type-v q8_0
	--cache-ram 4096
	--defrag-thold 0.1
	--no-warmup
    )
fi

exec "$SCRIPT_DIR/build/bin/llama-cli" "${args[@]}" "$@"
