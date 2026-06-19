# run.sh

Starts `llama-server` (or `llama-diffusion-server`) with pre-tuned settings for the available models on this machine. All servers listen on localhost; any extra arguments are passed through to the binary.

## Quick reference

```
./run.sh [--gemma-mtp] [--qwen3] [--diffusion] [--vulkan] [extra llama-server flags...]
```

## Flags

| Flag | Effect |
|------|--------|
| *(none)* | Gemma-4 26B MoE, multi-modal, Q8_0 — default |
| `--gemma-mtp` | Gemma-4 26B MoE Q8_0 + MTP speculative decoding |
| `--qwen3` | Qwen3-Coder-Next Q8_0 (~80B) |
| `--diffusion` | DiffusionGemma 26B Q8_0, starts `llama-diffusion-server` |
| `--vulkan` | Force Vulkan backend (`HIP_VISIBLE_DEVICES=""`) instead of ROCm |

Flags can be combined: `--diffusion --vulkan` runs the diffusion server on the Vulkan backend.

## Paths and ports

| Config | Binary | Port | Model |
|--------|--------|------|-------|
| default | `llama-server` | 8080 | `gemma-4-26B-A4B-it-Q8_0.gguf` + mmproj |
| `--gemma-mtp` | `llama-server` | 8080 | same + `gemma-4-26B-A4B-it-assistant-Q8_0.gguf` |
| `--qwen3` | `llama-server` | 8080 | `Qwen_Qwen3-Coder-Next-Q8_0-*.gguf` |
| `--diffusion` | `llama-diffusion-server` | 8190 | `diffusiongemma-26B-A4B-it-Q8_0.gguf` |

## Mode details

### Default — Gemma-4 26B multi-modal
```
./run.sh
./run.sh --vulkan
```
Starts `llama-server` on port 8080 with the vision encoder loaded and offloaded to GPU (`--mmproj-offload`), enabling image inputs. KV cache is quantized to Q8_0 to save VRAM. Context: 131072 tokens.

### `--gemma-mtp` — Gemma-4 with speculative decoding
```
./run.sh --gemma-mtp
./run.sh --gemma-mtp --vulkan
```
Loads the main Gemma-4 26B model plus the `gemma4-assistant` draft model for multi-token prediction (`--spec-type draft-mtp --spec-draft-n-max 3`). The assistant reads frozen K/V from the target's KV cache — **do not add `--cache-type-k/v`** here; quantizing the KV tensors breaks the frozen-KV attention and collapses draft acceptance to ~0%. Reasoning is enabled in `auto` mode. ~45% faster than default at 256-token outputs.

### `--qwen3` — Qwen3-Coder-Next (~80B)
```
./run.sh --qwen3
```
Vulkan-only in practice (model is too large for ROCm on this machine). KV cache quantized to Q8_0; `--cache-ram 4096` offloads idle slot state to RAM. Reasoning in `auto` mode.

### `--diffusion` — DiffusionGemma entropy-bound
```
./run.sh --diffusion
./run.sh --diffusion --vulkan
```
Starts `llama-diffusion-server` on port **8190** (not 8080). Exposes `GET /health` and `POST /v1/chat/completions` (OpenAI-compatible, streaming and non-streaming). The `--device ROCm0` / `--device Vulkan0` argument is injected automatically based on whether `--vulkan` is set — this prevents the Strix Halo iGPU from being double-counted through both backends.

## Backend selection

Without `--vulkan`, ROCm is the default GPU backend.  
With `--vulkan`, `HIP_VISIBLE_DEVICES=""` is set before exec, hiding all HIP devices so the Vulkan backend is used exclusively.

Benchmark (Radeon 8060S, 256-token output):

| Config | ROCm | Vulkan |
|--------|------|--------|
| diffusion | 28.3 t/s | 29.0 t/s |
| gemma-mtp | 52.7 t/s | 59.5 t/s |
| default | 37.0 t/s | 41.0 t/s |
| qwen3 | — | 36.1 t/s |

## Passing extra flags

Any unrecognized argument is forwarded directly to the binary:

```bash
# increase context to 256k for the default Gemma path
./run.sh -c 262144

# run on a different port
./run.sh --gemma-mtp --port 9090

# verbose logging
./run.sh --diffusion --vulkan --verbose
```
