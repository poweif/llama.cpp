#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 'a' is just a placeholder, setup-env.sh expects an argument for some reason
source ~/VulkanSDK/1.4.350.0/setup-env.sh 'a'
CXX=/usr/bin/clang++-17 CC=/usr/bin/clang-17 cmake -B "$SCRIPT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_SERVER=ON \
    "$SCRIPT_DIR"

cmake --build "$SCRIPT_DIR/build" --target llama-server llama-cli -j"$(nproc)"
