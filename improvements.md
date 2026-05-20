# Inference Performance & Memory Efficiency Improvements

Analysis of `src/`, `ggml/src/`, and `common/`, cross-referencing TODOs, FIXMEs, and
architectural patterns. Items are grouped by impact area and ordered roughly by how much
work they entail vs. how much they would help for large-model inference.

---

## 1. KV Cache: Hadamard Rotation Matrix — Host→Device Transfer on Every Forward Pass

**Where:** `src/llama-kv-cache.cpp:306`

When quantized KV types (`type_k`, `type_v`) are used with attention rotation (required for
quantized caches), the Hadamard rotation matrices are kept in **host memory** and transferred
to device on every forward pass. For large models with many layers, this is a repeated
host→device PCIe transfer that blocks the GPU.

**Proposal:** Pre-upload the rotation matrices to each backend's buffer type at cache
construction time. The matrices are fixed for the lifetime of the model; a one-time upload
eliminates recurring transfers and removes this GPU pipeline stall.

---

## 2. KV Cache Rotation: Asymmetric Matrix Sizing for K is Unresolved

**Where:** `src/llama-kv-cache.cpp:1320–1325`

For V, the code deliberately uses a **64×64** rotation matrix rather than the maximum possible
size, because empirically this is better for V. For K, the code still uses the maximum-size
matrix. There is an open TODO (with a linked PR discussion) noting that the smallest matrix
might also be beneficial for K, and this was never investigated and implemented.

**Proposal:** Benchmark and, if confirmed, use the same 64×64 (or smallest valid) rotation
matrix for K. This directly reduces the memory occupied by the rotation tensors and the flop
count for the rotation operation on every attention layer.

---

## 3. Multi-Stream KV Cache: `equal_seqs` Path is Blocked

**Where:** `src/llama-kv-cache.cpp:1651–1655`, `src/llama-graph.cpp:159`, `src/llama-graph.cpp:2123`

Three places hard-abort with `GGML_ASSERT(!ubatch->equal_seqs())` when `n_stream > 1`. The
`equal_seqs` batching path (where all sequences have the same length, enabling tighter packing)
is incompatible with the multi-stream KV cache layout. This forces the system to fall back to
`split_simple` for any multi-stream scenario, which is less efficient.

**Proposal:** Implement the `ubatch.n_seqs_unq`-based splitting mentioned in
`llama-graph.cpp:2123`. When sequences are equal length, Q/K/V tensors can be streamed
independently per sequence. This unlocks better parallelism for multi-sequence inference,
important for server batch workloads.

---

## 4. `get_rows` CPU Threading: Intentionally Disabled

**Where:** `ggml/src/ggml-cpu/ggml-cpu.c:2307–2310`

`GET_ROWS` (embedding lookups) is hard-coded to `n_tasks = 1`, bypassing the thread pool,
with a comment that "the cost of launching threads decreases performance with GPU offloading."
For CPU-only large models or split-device setups where many embeddings are on CPU, this is
a bottleneck.

**Proposal:** Make the thread count conditional: use `n_tasks = 1` only when a GPU backend is
active for this buffer, and `n_tasks = n_threads` otherwise. The logic is already well-understood
and the change is low-risk.

---

## 5. Fused Op Detection: Runs Per-Node at Compute Time, Not at Plan Time

**Where:** `ggml/src/ggml-cpu/ggml-cpu.c:3053`

On the CPU, every forward pass re-runs `ggml_cpu_try_fuse_ops()` for every node at compute
time — fused-op pattern matching happens redundantly on every inference call. The TODO
acknowledges this should be moved into `ggml_graph_plan`, where it would run once per graph
reservation and cache the fusion decisions.

**Proposal:** Move fused-op detection into the graph planning phase. The decisions are cached
for the lifetime of the reserved graph, eliminating repeated pattern-matching overhead on the
hot path. This is especially significant for long context (many layers × many tokens) where
the compute loop executes thousands of nodes.

---

## 6. KV Cache Session Restore: `apply_ubatch()` Does Not Handle `kv_cell_ext`

**Where:** `src/llama-kv-cache.cpp:2119`

When restoring from a saved KV cache session, `llama_kv_cell_ext` metadata (used in the
extended-streaming KV layout) cannot be restored because `apply_ubatch()` does not support it.
This means session state is partially lost on reload, requiring a cold re-evaluation of the
context.

**Proposal:** Implement `kv_cell_ext` serialization and restoration in `apply_ubatch`. For
users running long-context sessions (exactly the scenario where you want the largest model
fitting in memory), this avoids wasting KV compute budget re-warming after restarts.

---

## 7. Batch Tensor Extraction (`tensor_get`) During Session Save is Serial

**Where:** `src/llama-context.cpp:2384`

During KV cache state serialization, `llama_io_write_host::~llama_io_write_host` calls
`ggml_backend_tensor_get()` sequentially for each tensor in `winfos`. For large models with
large KV caches, this is a blocking serial transfer from device.

**Proposal:** Add a batched backend tensor-get API. For CUDA/ROCm/Vulkan backends, multiple
memcpy operations can be pipelined or issued concurrently. Even a simple loop using async
device copies with a single synchronize at the end would be substantially faster than
issuing-and-waiting per tensor.

---

## 8. Sequence Padding Split Mode for Encoders

**Where:** `src/llama-context.cpp:1331`

Encoder models always use `split_simple` (sequential token split), even though padded
splitting would allow `ubatch.equal_seqs == true`, enabling more efficient attention graph
construction. The TODO explicitly notes this.

**Proposal:** Add a `split_padded` mode that pads variable-length sequences to the same length
within a micro-batch, enabling the `equal_seqs` fast path through the attention code. This is
relevant for embedding-model workloads and cross-attention in encoder-decoder models (T5, BART).

---

## 9. Flash Attention: Auto-Detection Mismatch with `--no-kv-offload`

**Where:** `src/llama-context.cpp:473`

The FA auto-detection logic (marked `FIXME`) misidentifies device mismatches when
`--no-kv-offload` is used, causing FA to be **incorrectly disabled** in this config. For large
models where the user explicitly wants KV on CPU and compute on GPU (to fit the model), this
silently disables a major throughput optimization.

**Proposal:** Fix the device-mismatch check to correctly compare the FA tensor's compute device
against the actual KV storage device (not the layer's default device). The `--no-kv-offload`
path should be treated as a deliberate host-KV configuration, not a mismatch, so FA remains
enabled.

---

## Summary

| # | Area | Impact | Effort | File |
|---|------|--------|--------|------|
| 1 | KV rotation matrix: host→device transfer | High (every forward pass, all layers) | Low | `llama-kv-cache.cpp:306` |
| 9 | FA wrongly disabled with `--no-kv-offload` | High (large model users) | Low | `llama-context.cpp:473` |
| 3 | `equal_seqs` multi-stream path blocked | Medium (multi-sequence server use) | High | `llama-kv-cache.cpp:1651`, `llama-graph.cpp:159` |
| 5 | Fused op detection at compute time | Medium (CPU inference, long context) | Medium | `ggml-cpu.c:3053` |
| 7 | Serial tensor_get during session save | Medium (large KV cache saves) | Medium | `llama-context.cpp:2384` |
| 2 | K rotation matrix asymmetry | Low–medium (quantized KV) | Low | `llama-kv-cache.cpp:1320` |
| 4 | `get_rows` single-threaded on CPU | Low (CPU-heavy workloads) | Low | `ggml-cpu.c:2307` |
| 6 | KV session restore incomplete | Low (session-reload latency) | Medium | `llama-kv-cache.cpp:2119` |
| 8 | No padded encoder split mode | Low (encoder models) | Medium | `llama-context.cpp:1331` |

The highest-leverage items for fitting and running the largest possible model are **#1**
(removes PCIe stalls on every inference step when using quantized KV) and **#9** (restores FA
for partial-offload configs).

---

# Kernel-Level Improvements

Analysis of the actual compute kernels in `ggml/src/ggml-cpu/`, `ggml/src/ggml-vulkan/`, and
the flash attention implementation. Findings are split into CPU, Vulkan, and attention sections.

---

## CPU Kernels

### K1. F32 Dot Product Has No AVX2 Path

**Where:** `ggml/src/ggml-cpu/vec.cpp:11–230`

`ggml_vec_dot_f32` has hand-written paths for ARM SVE and RISC-V RVV, but **no AVX2
implementation**. x86_64 systems that lack AVX-512 fall back to scalar. This is the
innermost loop for F32 matrix multiplication — the operation executed most in non-quantized
inference.

**Proposal:** Add an AVX2 path with 8-element SIMD and 4-accumulator unrolling (32 elements
per iteration). This is the single highest-impact missing optimization for x86_64 CPU
inference.

---

### K2. Multi-Row vec_dot (MMLA Path) Disabled for Odd Dimensions

**Where:** `ggml/src/ggml-cpu/ggml-cpu.c:1432–1434`

The `num_rows_per_vec_dot` path processes two output rows simultaneously using MMLA-style
instructions (2x throughput on ARM, and analogous multi-row kernels on x86). A boundary check
forces `num_rows_per_vec_dot = 1` when the remaining row count is not a multiple of the
per-call row count. This disables the fast path for the majority of non-power-of-2 matrix
dimensions, which includes almost all real attention head sizes.

**Proposal:** Allow partial multi-row dispatch: process trailing rows individually after
exhausting full multi-row tiles. The boundary check can be moved to inside the loop rather
than disabling the accelerated path globally.

---

### K3. Matrix Mul Tile Size is Hardcoded

**Where:** `ggml/src/ggml-cpu/ggml-cpu.c:1193–1194`

`blck_0 = 16, blck_1 = 16` is hardcoded. The value does not adapt to the CPU's L1/L2 cache
size or SIMD width. `CACHE_LINE_SIZE_F32` is already defined in `ops.h` but is not used for
tiling decisions. On modern x86 with 256-bit AVX2, an optimal tile might be 32×32 (fits two
cache lines per row, keeps accumulators in register).

**Proposal:** Compute tile sizes at plan time from `CACHE_LINE_SIZE_F32` and the SIMD width
detected at startup, and cache the result in `ggml_graph_plan`. Pairs naturally with the
fused-op detection move described in improvement #5.

---

### K4. Type Conversion Not Pipelined with Computation

**Where:** `ggml/src/ggml-cpu/ggml-cpu.c:1313–1355`

When `src1->type` does not match `vec_dot_type`, input activations are quantized into
`params->wdata` in a separate phase before any dot products begin. All threads synchronize
at a barrier (line 1355) before compute starts, so the slowest thread's quantization time
blocks the entire pool.

**Proposal:** Double-buffer: while threads are computing dot products for tile N, quantize
tile N+1 in the background. The barrier can then be replaced with a per-tile signal between
the quantization and compute phases, hiding memory latency behind arithmetic.

---

### K5. Q4_K / Q5_K / Q6_K Generic Dequantization is Redundant

**Where:** `ggml/src/ggml-cpu/quants.c:645–853`

The generic Q4_K dot-product function dequantizes the entire block (256 elements) into a
256-byte stack buffer (`aux8[QK_K]`) before doing any multiply-accumulate. This is a
two-pass approach: dequantize, then dot. The double traversal doubles the memory traffic and
the stack pressure. Q5_K adds a second pass for high bits; Q6_K requires four bit extractions
per 32-element chunk.

**Proposal:** Integrate dequantization into the dot-product loop — dequantize a small tile
(e.g., 32 elements) directly into a SIMD register, multiply-accumulate, discard. This halves
memory traffic and eliminates the stack buffer. The x86 AVX2 path for Q4_K (`arch/x86/`)
already does this with `_mm256_maddubs_epi16`; the generic path should follow the same
pattern.

---

### K6. IQ Series Lookup Tables Are Scalar in the Hot Loop

**Where:** `ggml/src/ggml-cpu/arch/x86/quants.c:2535–2575` (IQ2_xxs)

IQ2/IQ3 quantization types decode weights through a compressed grid lookup:
`iq2xxs_grid[aux8[l]]`. This gather operation — one table lookup per 8-element group — is
performed inside the SIMD loop but cannot be vectorized as written, because the indices are
scalar. The surrounding AVX2 accumulators then stall waiting for each scalar lookup.

**Proposal:** Pre-expand the lookup table entries for a block into a contiguous buffer before
entering the SIMD accumulation loop, or use `_mm256_i32gather_epi32` (AVX2 gather) to load
8 entries simultaneously. The grid is small enough (2048 entries × 8 bytes = 16 KB) to
remain in L1 cache during inference.

---

### K7. RMS Norm Sum-of-Squares is Scalar

**Where:** `ggml/src/ggml-cpu/ops.cpp:3757–3764`

The sum-of-squares loop in `ggml_compute_forward_rms_norm_f32` is a simple scalar
accumulator — there is even a comment asking whether it is worth switching to SIMD (line
3758). RMS norm runs on every transformer layer before both attention and FFN, making it a
frequently executed operation.

**Proposal:** Vectorize the variance computation with SIMD horizontal sum (AVX2:
`_mm256_dp_ps` or manual `hadd`). The existing fused `rms_norm + mul` path in
`ggml_compute_forward_rms_norm_mul_fused` would benefit from the same change.

---

### K8. Softmax Mask Application is Scalar with F16 Conversion

**Where:** `ggml/src/ggml-cpu/ops.cpp:5345–5357`

The attention mask addition (`wp[i] += slope * mp_f16[i]`) is a scalar loop with an
`F16→F32` conversion on every element. This runs for every row of the attention matrix —
for a 4096-token context with 32 heads that is ~134 million scalar operations per layer.

**Proposal:** Vectorize with `_mm256_cvtph_ps` (AVX2 F16→F32 conversion) and a fused
multiply-add. On ARM, `vcvt_f32_f16` + `vfma` achieves the same. The slope broadcast can be
hoisted out of the loop as a SIMD constant.

---

### K9. x86 Tensor Repacking Uses Generic Fallback

**Where:** `ggml/src/ggml-cpu/repack.cpp` and `arch/x86/` directory

ARM and RISC-V have architecture-specific repack routines for Q8_K (NEON and RVV variants).
x86 falls back to the generic implementation. Repacking — converting tensors from the
file-format interleaving into the SIMD-friendly 4×4 or 4×8 layout — runs on every new
prompt or when the KV cache is populated, so it matters for time-to-first-token.

**Proposal:** Add AVX2 repack kernels mirroring the ARM NEON variants. The transformation is
a structured byte-shuffle that maps directly to `_mm256_shuffle_epi8` (pshufb) instructions.

---

## Vulkan Kernels (AMD Ryzen AI Strix Halo)

### V1. Flash Attention Falls Back to Scalar on AMD RDNA

**Where:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp:3087–3088`

The Vulkan backend has three flash attention code paths: `FA_COOPMAT2` (NVIDIA-only NV
extension), `FA_COOPMAT1` (VK_KHR_cooperative_matrix), and `FA_SCALAR`. AMD RDNA does not
support either cooperative matrix extension, so **all AMD hardware uses FA_SCALAR**. The
scalar path uses `Br = 1` (single query row per workgroup) with full global memory reads of
K and V on every block iteration. This is ~3-4× slower than the coopmat paths.

**Proposal:** Implement a Vulkan FA path that uses subgroup operations
(`subgroupClusteredMax`, `subgroupClusteredAdd`) with larger `Br` tiles (e.g., Br=8 or 16)
and explicit shared memory staging of K/V blocks. This does not require cooperative matrix
extensions and would close the majority of the gap on RDNA hardware.

**Validation:**
1. Confirm the current code path: temporarily add `GGML_LOG_INFO` at
   `ggml-vulkan.cpp:3087–3088` to log which `FaCodePath` is selected at context init.
   On Strix Halo this must print `FA_SCALAR`.
2. After implementing the subgroup-tiled path, re-run the log and confirm the new path
   is selected for RDNA3.
3. Correctness: run `llama-cli` with a small model (e.g. Qwen2.5-0.5B-Instruct) and
   compare per-token logits between the scalar and new path using `--logits-all`. Values
   must agree to within F16 rounding error (~0.1% relative).
4. Performance: run `llama-bench -m <model> -ngl 99 -pg 512,1` before and after. Expect
   meaningful improvement in prompt-processing speed (pp512) on long contexts.
5. Stress-test varying head sizes: run with models that have `n_embd_head` of 64, 128, and
   256 (e.g. Llama-3 = 128, Phi-3 = 96) to ensure the Br tiling is correct for each.
6. Test with GQA models (DeepSeek, Llama-3) where `n_kv_head != n_head`; verify the
   subgroup reduction handles the GQA broadcast correctly.

---

### V2. K/V Shared Memory Staging Disabled on AMD

**Where:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp:3002, 3053`

```cpp
result.shmem_staging = (vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;
```

The `SHMEM_STAGING` optimization — loading K and V blocks into shared memory before the
inner dot loop — is gated to NVIDIA. On AMD, K and V are read repeatedly from global memory
on each flash attention iteration. RDNA's shared memory (LDS) is 65 KB per workgroup, large
enough for typical head sizes (128 or 256 dimensions at F16).

**Proposal:** Enable `shmem_staging` for AMD when `hsk * Bc * sizeof(f16) < LDS_budget`.
The guard condition already exists; it just needs the NVIDIA vendor check removed and replaced
with a capacity check against `maxComputeSharedMemorySize`.

**Validation:**
1. Before the change, add a temporary log at `ggml-vulkan.cpp:3002` to confirm
   `shmem_staging = 0` on Strix Halo.
2. Compute the expected LDS requirement manually for the target model:
   `hsk * Bc * 2` bytes for F16 K, same for V. For hsk=128, Bc=32: 128×32×2×2 = 16 KB.
   Confirm this is below `VkPhysicalDeviceLimits::maxComputeSharedMemorySize` (query with
   `vulkaninfo | grep maxComputeSharedMemory`).
3. After enabling, confirm `shmem_staging = 1` is logged for the model under test.
4. Run `rocprof --stats llama-bench ...` before and after and compare
   `L2CacheHit` and `MemUnitBusy` counters. Staging should increase L2 hit rate and reduce
   raw memory bandwidth consumed by the FA kernel.
5. Correctness: identical logit comparison as in V1 step 3.
6. Test with hsk=256 (Falcon, some Yi variants) to confirm the capacity guard correctly
   disables staging when LDS would be exceeded.

---

### V3. UMA (Integrated APU) Zero-Copy Not Implemented

**Where:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp:5934`

The Vulkan backend correctly detects integrated/UMA GPUs (`deviceType == eIntegratedGpu`)
but does not change its memory allocation strategy. On Ryzen AI Strix Halo, CPU and GPU
share the same physical memory — there is no PCIe bus. Despite this, the code still:
- Allocates device-local buffers when available (a hint that triggers a separate allocation
  from shared RAM)
- Uses a transfer queue to "copy" between host and device (a no-op on true UMA but still
  incurs driver overhead)
- Never uses `HOST_VISIBLE | DEVICE_LOCAL` coherent memory, which on APUs maps to the
  true zero-copy shared pool

**Proposal:** On UMA devices, prefer `eHostVisible | eDeviceLocal | eHostCoherent` memory
for all tensors that are read from CPU (model weights during initial load, input embeddings).
This eliminates the copy entirely. The `uma` flag is already computed and passed around; it
just needs to influence the `vk::MemoryPropertyFlags` priority list in the allocator.

**Validation:**
1. Confirm UMA detection fires: `uma` is set at `ggml-vulkan.cpp:5934`. Log its value at
   startup — must be `true` on Strix Halo (`deviceType == eIntegratedGpu`).
2. Before the change, use `vulkan-dumper` or add a temporary log in the allocator to record
   which `vk::MemoryType` index each weight tensor lands in. Device-local-only memory
   (no `HOST_VISIBLE` bit) confirms the problem.
3. After the change, repeat step 2 and confirm tensors land in a heap with both
   `DEVICE_LOCAL` and `HOST_VISIBLE` bits set. On Strix Halo this maps to the APU's unified
   memory pool.
4. Measure model load time with `time llama-cli --no-warmup ...` before and after.
   The copy from mmap'd file into device-local memory should be eliminated, reducing load time
   roughly in proportion to model size.
5. Measure peak RSS (`/usr/bin/time -v`) before and after. Correct UMA allocation should
   eliminate the double-buffer (file mapping + device copy), reducing peak RAM by ~model size.
6. Correctness: full generation run comparing output text (sampling with fixed seed) before
   and after to confirm weights are identical.
7. Run the existing ggml backend tests (`./bin/test-backend-ops -b Vulkan`) to confirm no
   regressions in tensor read/write paths.

---

### V4. Quantized Matmul `BK_STEP` Not Tuned for RDNA

**Where:** `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp:87–92`

`BK_STEP = 4` for dense quantized matmul is a fixed constant. RDNA has 65 KB of LDS.
For Q5_K (36 bytes/block): `buf_a = 64 * 4 * 36 = 9 KB`, `buf_b = 64 * 4 * 16 = 4 KB` —
total ~13 KB, well under the 65 KB limit. Doubling to `BK_STEP = 8` would improve
compute-to-memory overlap at the cost of 26 KB LDS, still safely within budget.

**Proposal:** Expose `BK_STEP` as a specialization constant (like `BM`, `BN`, `WM`, `WN`
already are) and set it per device and per quantization type in `ggml-vulkan.cpp` during
pipeline creation. RDNA should use BK_STEP=8 for Q4_K/Q5_K.

**Validation:**
1. Compute LDS usage at BK_STEP=8 for each quant type before touching any code:
   - Q4_K: `buf_a = 64 * 8 * sizeof(block_a_cache_Q4K)` + `buf_b = 64 * 8 * sizeof(block_b_cache)`.
     Confirm total < 65536 bytes (RDNA LDS limit).
   - Q5_K: same calculation with `block_a_cache_Q5K` (8 int32s + dm = 36 bytes).
   - If either exceeds budget, cap at the largest safe BK_STEP.
2. After exposing `BK_STEP` as a specialization constant, inspect the compiled SPIR-V with
   `spirv-dis` and grep for the `OpSpecConstant` declaration to confirm it is being set.
3. Benchmark prefill throughput: `llama-bench -m <Q4_K model> -ngl 99 -pg 512,1 -pg 2048,1`
   at BK_STEP=4 (baseline) and BK_STEP=8. The pp metric should improve on Strix Halo.
4. Use `rocprof --stats` to compare `LDSBankConflict` and `MemUnitBusy` counters between the
   two settings. Higher BK_STEP should reduce global memory pressure.
5. Correctness: compare matmul output against a reference F32 computation using
   `test-backend-ops -b Vulkan -o MUL_MAT` — ensure max absolute error stays within the
   expected quantization tolerance for the quant type.
6. Test at batch sizes 1, 4, 16, 512 to confirm the improvement holds across decode and
   prefill regimes and does not regress the decode (pp=1) case.

---

### V5. RDNA Subgroup Tuning Not Updated for RDNA2/RDNA3

**Where:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp:3302–3306`

Pipeline workgroup sizes are tuned per RDNA generation:
```cpp
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = { ... };
```
RDNA1-specific overrides exist (64-wide subgroups for softmax, argmax, mul_mat_vec), but
there are no equivalent `rdna2_pipelines` or `rdna3_pipelines` maps. RDNA2 and RDNA3
differ in wave32 vs wave64 defaults and in how dual-issue ALU affects optimal workgroup size.
Strix Halo is RDNA3.5 and falls through to defaults.

**Proposal:** Add RDNA2/RDNA3 specialization tables. At minimum, `mul_mat_vec` for F16
should use 32-wide subgroups on RDNA2+ (wave32 is the native mode, wave64 has dual-issue
overhead).

**Validation:**
1. Identify Strix Halo's architecture enum value: add a log at device init that prints
   `device->architecture`. Confirm it resolves to `AMD_RDNA3` (or whichever enum covers
   RDNA3.5). If it falls through to `OTHER`, the architecture detection itself needs fixing
   first using the device's PCI device ID.
2. Add a log in the pipeline-creation path to record which subgroup size is being selected
   for `mul_mat_vec_f16` before and after adding the RDNA3 table. Before: default (likely 64).
   After: 32.
3. Benchmark `mul_mat_vec` decode throughput: `llama-bench -m <F16 or Q8_0 model> -ngl 99
   -p 0 -n 128`. This exercises the decode path heavily. Compare tokens/sec at wave32 vs wave64.
4. Use `rocprof --stats` on the decode benchmark and inspect `VALUUtilization` and
   `VALUBusy`. Wave32 on RDNA3 should yield higher utilization for this workload.
5. Run the same benchmark for `mul_mat_vec` with Q4_K and Q5_K (quantized decode) to ensure
   the wave32 setting does not degrade those paths, which may have different occupancy
   characteristics.
6. Run `test-backend-ops -b Vulkan -o MUL_MAT_VEC` to check numerical correctness is
   unaffected by the subgroup size change.

---

### V6. IQ Series Lacks Quantized Matmul Shaders

**Where:** `ggml/src/ggml-vulkan/vulkan-shaders/` directory

IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS have `mul_mat_vec` shaders but **no `mul_mmq`
shader** (the tiled matmul path used for prompt processing). For batch sizes > 1 or during
prefill, inference falls back to dequantize-then-matmul (two passes). IQ4_NL/IQ4_XS also
have no flash attention integration.

**Proposal:** Implement `mul_mmq` variants for the IQ series. IQ2/IQ3 use lookup tables;
the Vulkan equivalent would use `texelFetch` or a SSBO lookup table in shared memory. Given
that IQ types are the primary way to fit larger models in memory, this is directly relevant
to the stated goal.

**Validation:**
1. Confirm the fallback is active: add a temporary log in `ggml-vulkan.cpp` where the matmul
   dispatch selects a pipeline. For IQ2/IQ3 types with batch > 1, it must currently log that
   it is falling back to the dequantize-then-matmul path rather than `mul_mmq`.
2. For a reference output, run `llama-bench -m <IQ2_XXS model> -ngl 99 -pg 512,1` before the
   change and record pp512. This is the baseline dequant+matmul prefill speed.
3. After implementing the `mul_mmq` shader, run the same benchmark and compare pp512.
   Expect a meaningful improvement since two full passes over the weight data are reduced to one.
4. Correctness: run `test-backend-ops -b Vulkan -o MUL_MAT` with an IQ2_XXS and IQ3_S weight
   tensor. Compare output against the reference dequant+F32 matmul result; max absolute error
   should be within the quantization error floor for those types (IQ2 ≈ 2–3% relative error,
   same as the CPU reference).
5. Test the lookup table path specifically: the SSBO/shared-memory grid must produce the same
   8-element group values as `iq2xxs_grid[idx]` on CPU. Write a small standalone Vulkan compute
   test that runs the lookup on a known index set and compares against the CPU table.
6. Test at batch sizes 1 (should still use `mul_mat_vec`), 2, 8, 512 to confirm the dispatch
   threshold between `mul_mat_vec` and `mul_mmq` is correct for IQ types.
7. Run a full generation pass (`llama-cli -m <IQ2_XXS model> -p "Hello" -n 200`) and compare
   output text with a fixed seed against the dequant-path baseline to catch any correctness
   regression in the end-to-end pipeline.

---

## Flash Attention — CPU Inner Loop

### FA1. CPU Flash Attention Inner Loop Uses Un-vectorized Quant Kernels

**Where:** `ggml/src/ggml-cpu/ops.cpp:8338–8403`

The CPU flash attention implementation calls `kq_vec_dot()` for each key vector inside a
sequential token loop. `kq_vec_dot` dispatches to the same generic quantized dot-product
kernels analyzed above (Q4_K, Q5_K, IQ series) — which have no SIMD on the generic path.
The result is that attention computation, the O(N²) portion of inference, runs entirely
scalar on CPU for quantized models.

**Proposal:** Write a fused CPU flash attention kernel that inlines the quantized dot product
directly, enabling the compiler to vectorize the dequantize + multiply-accumulate + online
softmax sequence as a single loop body. The existing GPU flash attention tile structure
(K-tile × Q-tile) should be mirrored.

---

### FA2. No Block Tiling for K/V on CPU Flash Attention

**Where:** `ggml/src/ggml-cpu/ops.cpp:8338`

The CPU flash attention iterates over every key token sequentially. There is no tiling over
the KV cache — each key vector is fetched from memory individually. This produces poor cache
behavior for long contexts: a 32K-token context with 128-dim keys means 32K separate cache
line loads per query.

**Proposal:** Tile the key dimension: load a block of Bc=64 keys into L1 cache, compute all
K·Q scores for the block, update the online softmax, then accumulate into V. This matches
what `ggml_compute_forward_flash_attn_ext_tiled` (line 8452) partially implements for F16,
but that path does not handle quantized K types.

---

## Kernel Summary

| ID | Area | Scope | Impact | Effort |
|----|------|-------|--------|--------|
| K1 | AVX2 F32 dot product missing | CPU x86 | High | Low |
| V1 | Scalar flash attention on AMD RDNA | Vulkan | High | High |
| V3 | UMA zero-copy not used on APU | Vulkan | High | Medium |
| FA1 | Flash attn inner loop is un-vectorized | CPU | High | High |
| K5 | Q4_K/Q5_K/Q6_K two-pass dequant | CPU | Medium | Medium |
| K6 | IQ lookup tables scalar in SIMD loop | CPU x86 | Medium | Medium |
| V2 | K/V shmem staging disabled on AMD | Vulkan | Medium | Low |
| FA2 | No KV block tiling on CPU FA | CPU | Medium | High |
| K2 | Multi-row vec_dot disabled odd dims | CPU | Medium | Low |
| K7 | RMS norm sum-of-squares is scalar | CPU | Medium | Low |
| K8 | Softmax mask loop scalar + F16 conv | CPU | Medium | Low |
| V4 | BK_STEP hardcoded, not tuned RDNA | Vulkan | Medium | Low |
| K3 | Tile size hardcoded, not cache-adaptive | CPU | Low–Med | Medium |
| K4 | Quant phase blocks compute phase | CPU | Low–Med | Medium |
| V5 | RDNA2/3 subgroup sizes not tuned | Vulkan | Low | Low |
| V6 | IQ types lack mul_mmq shader | Vulkan | Low | High |
| K9 | x86 tensor repack uses generic path | CPU | Low | Medium |
