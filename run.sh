#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

use_vulkan=0
use_qwen3=0
passthrough=()
for arg in "$@"; do
    case "$arg" in
        --vulkan) use_vulkan=1 ;;
        --qwen3)  use_qwen3=1 ;;
        *)        passthrough+=("$arg") ;;
    esac
done
set -- "${passthrough[@]+"${passthrough[@]}"}"

if (( use_vulkan )); then
    export HIP_VISIBLE_DEVICES=""
fi

if (( use_qwen3 )); then
    args=(
	-m /home/poweif/models/Qwen_Qwen3-Coder-Next-Q6_K_L-00001-of-00002.gguf
	-ngl 99          # offload all layers to GPU
	-fa on           # flash attention: faster, lower VRAM on long contexts
	-c 131072        # max context length (tokens)
	-n 16384         # max tokens to generate per request
	-b 4096          # prompt batch size: larger = faster ingestion, more VRAM
	-ub 512          # micro-batch size: smaller = lower latency per decode step
	--no-context-shift  # error on context overflow instead of silently rotating the window
	--cache-reuse 256   # reuse KV cache for requests sharing a 256-token prefix (system prompt, file context)
	--jinja          # enable Jinja2 chat templates
	--port 8080
	--cache-type-k q8_0  # quantize KV cache to Q8_0: lower VRAM, minimal quality loss
	--cache-type-v q8_0
	# --cache-idle-slots is on by default but silently no-ops without --cache-ram.
	# With it active, KV state from finished conversations is offloaded to RAM and
	# freed from the main buffer, preventing unbounded memory growth over time.
	--cache-ram 4096
	# Trigger KV cache defragmentation when 10% of cells are fragmented.
	# Deprecated flag but still functional; reduces fragmentation-driven bloat.
	--defrag-thold 0.1
    )

else
    #    -m /home/poweif/models/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf
    #     -m /home/poweif/models/Qwen3-32B-Q4_K_M.gguf
    #    -m /home/poweif/models/Qwen3-72B-Instruct.Q4_K_M.gguf \
    #    -m /home/poweif/models/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf \
    args=(
	-m /home/poweif/models/gemma-4-31B-it-Q8_0.gguf
	-ngl 99          # offload all layers to GPU
	-fa on           # flash attention: faster, lower VRAM on long contexts
	-c 262144        # max context length (tokens)
	-n 16384         # max tokens to generate per request
	-b 4096          # prompt batch size: larger = faster ingestion, more VRAM
	-ub 512          # micro-batch size: smaller = lower latency per decode step
	--no-context-shift  # error on context overflow instead of silently rotating the window
	--cache-reuse 256   # reuse KV cache for requests sharing a 256-token prefix (system prompt, file context)
	--jinja          # enable Jinja2 chat templates (required for Gemma's format)
	--port 8080
	--cache-type-k q8_0  # quantize KV cache to Q8_0: lower VRAM, minimal quality loss
	--cache-type-v q8_0
	# --cache-idle-slots is on by default but silently no-ops without --cache-ram.
	# With it active, KV state from finished conversations is offloaded to RAM and
	# freed from the main buffer, preventing unbounded memory growth over time.
	--cache-ram 4096
	# Trigger KV cache defragmentation when 10% of cells are fragmented.
	# Deprecated flag but still functional; reduces fragmentation-driven bloat.
	--defrag-thold 0.1
    )
fi

exec "$SCRIPT_DIR/build/bin/llama-server" "${args[@]}" "$@"
