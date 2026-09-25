# High-Performance CUDA Kernel Engineering & Execution Framework

[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C++20](https://img.shields.io/badge/Standard-C%2B%2B20-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Modal](https://img.shields.io/badge/Platform-Modal%20GPU-000000?logo=modal&logoColor=white)](https://modal.com)
[![Architecture](https://img.shields.io/badge/Compute-sm__75%20%7C%20T4-green)](https://www.nvidia.com/en-us/data-center/tesla-t4/)

A systems research and engineering laboratory for designing, profiling, and benchmarking high-performance GPU kernels. This repository explores arithmetic intensity, warp-level execution patterns, memory hierarchy utilization (registers, shared memory SRAM, and global DRAM), vectorization via 128-bit memory instructions, and cloud-native serverless compilation workflows that decouple local development from remote NVIDIA accelerator execution.

---

## Architecture Overview

Modern deep learning workloads are frequently constrained by the GPU memory subsystem rather than peak compute throughput (FLOPs). This laboratory provides reference implementations and micro-benchmarks analyzing the transition across the Roofline model from bandwidth-bound kernels (Vector Add, Element-wise Activation) to compute-bound tiled matrix multiplication (GEMM).

```
+---------------------------------------------------------------------------------------+
|                                    Development Host                                   |
|   macOS (Apple Silicon arm64) | Homebrew GCC 16.2.0 | AST-level clangd Language Server |
+-------------------------------------------+-------------------------------------------+
                                            |
                                 Modal RPC / gRPC Tunnel
                                            |
+-------------------------------------------v-------------------------------------------+
|                           Remote Cloud Execution Target                               |
|          NVIDIA Tesla T4 (Turing Architecture, sm_75) | CUDA Toolkit 12.6             |
|                                                                                       |
|   +-----------------------+   +-----------------------+   +-----------------------+   |
|   |   Memory-Bound Base   |   |   Vectorized Memory   |   |   Cache-Tiled Compute |   |
|   |    Vector Addition    |   |    SiLU Activation    |   |     Matrix Multiply   |   |
|   |      (1D Stride)      |   |   (LD.E.128 float4)   |   |    (SRAM Shared Mem)  |   |
|   +-----------------------+   +-----------------------+   +-----------------------+   |
+---------------------------------------------------------------------------------------+
```

---

## Kernel Implementations & Architectural Analysis

### 1. Vectorized SiLU Activation with Hardware Alignment Handling (`cuda/silu.cu`, `cuda/silu.cpp`)

#### Mathematical Formulation
The Sigmoid Linear Unit ($\text{SiLU}$) activation function is defined as:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

#### Vectorized Memory Subsystem (`float4` / `LD.E.128`)
Pointwise activation functions are strictly memory-bandwidth bound. To maximize memory throughput, global memory loads are vectorized into 128-bit transactions using `float4`. This issues `LD.E.128` and `ST.E.128` assembly instructions, reducing the instruction issue bottleneck by $4\times$ and maximizing bus saturation compared to 32-bit scalar memory requests (`LD.E.32`).

#### The Alignment Problem in Non-Power-of-Two Geometries
Hardware `LD.E.128` instructions **strictly enforce 16-byte memory alignment**. In contiguous 2D row-major matrices with arbitrary column dimensions (e.g., $M = 128, N = 1027$):

$$\text{Address}(\text{row}, 0) = \text{base} + \text{row} \times N \times \text{sizeof}(\text{float})$$

For $N = 1027$:
- **Row 0**: $0 \times 4108\text{ B} = 0\text{ B} \equiv 0 \pmod{16}$ *(Aligned)*
- **Row 1**: $1 \times 4108\text{ B} = 4108\text{ B} \equiv 12 \pmod{16}$ *(Misaligned)*
- **Row 2**: $2 \times 4108\text{ B} = 8216\text{ B} \equiv 8 \pmod{16}$ *(Misaligned)*
- **Row 3**: $3 \times 4108\text{ B} = 12324\text{ B} \equiv 4 \pmod{16}$ *(Misaligned)*
- **Row 4**: $4 \times 4108\text{ B} = 16432\text{ B} \equiv 0 \pmod{16}$ *(Aligned)*

Naively executing `reinterpret_cast<float4*>(row_ptr)` across unaligned rows triggers a hardware **Misaligned Address Exception** (`XID 13 / XID 43`), aborting the CUDA context and leaving destination buffers zeroed.

#### Dual-Path Execution Strategy
The kernel implements dynamic runtime alignment verification with branch-free execution paths:

```cuda
unsigned int row = blockIdx.x;
float* row_ptr = matrix + row * n;
float4* matrix4 = reinterpret_cast<float4*>(row_ptr);

// Only execute 128-bit loads if row pointer satisfies 16-byte alignment
int vec_len = ((size_t)row_ptr % 16 == 0) ? (n / 4) : 0;

// Vectorized loop (runs for aligned rows)
for (int col = threadIdx.x; col < vec_len; col += blockDim.x) {
    float4 el = matrix4[col];
    matrix4[col] = make_float4(
        el.x / (1.0f + expf(-el.x)),
        el.y / (1.0f + expf(-el.y)),
        el.z / (1.0f + expf(-el.z)),
        el.w / (1.0f + expf(-el.w))
    );
}

// Remainder & unaligned scalar fallback loop
for (int col = vec_len * 4 + threadIdx.x; col < n; col += blockDim.x) {
    float x = row_ptr[col];
    row_ptr[col] = x / (1.0f + expf(-x));
}
```

---

### 2. Tiled Shared-Memory Matrix Multiplication (`cuda/tiled_matmuls.cu`)

#### Mathematical Formulation
Given matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the matrix product $C = A \times B$ computes:

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

#### Arithmetic Intensity & SRAM Caching
A naive matrix multiplication kernel accesses global DRAM for every multiply-accumulate operation ($O(N^3)$ global memory accesses for $2N^3$ FLOPs), yielding an operational intensity of $\approx 0.17 \text{ FLOP/byte}$, which bottlenecks on DRAM bandwidth.

The tiled kernel leverages on-chip **Shared Memory SRAM** (100+ TB/s aggregate bandwidth) to stage $T \times T$ sub-matrices:

$$\text{Memory Access Reduction Factor} = T$$

```cuda
__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // Collaborative cooperative staging from DRAM to Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads(); // Barrier: prevent warp execution before tile is fully populated

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads(); // Barrier: prevent overwriting tile while warps still compute
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

---

### 3. Coalesced Vector Addition (`cuda/vector_add.cu`)

Establishes the memory bus saturation baseline over $N = 2^{20}$ single-precision floats ($1\,048\,576$ elements). Demonstrates linear grid-stride index mapping:

$$i = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

Verifies strict host-side floating point tolerance and validates zero transaction divergence across warps.

---

## Serverless Cloud GPU Execution Pipeline (`cuda/modal_run.py`)

Local development on Apple Silicon arm64 cannot natively compile modern CUDA device code due to the absence of modern macOS NVIDIA toolchains. The orchestration engine dynamically compiles and runs kernels on remote cloud GPUs.

### Key Capabilities

1. **Content-Aware Translation Unit Compiler Dispatch**:
   The engine examines source AST and lexical markers (`__global__`, `cuda_runtime.h`, `cooperative_groups`, `<<<`, `cudaMalloc`, `cuda/pipeline`) to identify CUDA code regardless of extension:
   - **CUDA Source (`.cu` or CUDA `.cpp`)**: Compiled via `nvcc -std=c++17 -O2 -arch=native` (appending `-x cu` for `.cpp` files).
   - **Standard Host C++ (`.cpp`)**: Compiled via `g++ -std=c++20 -O2`.

2. **Native Architecture Specialization (`-arch=native`)**:
   Automatically detects the physical GPU attached to the container (e.g., Turing `sm_75` on Tesla T4). This allows modern CUDA C++ headers like `<cuda/pipeline>` and `<cooperative_groups.h>` (which require architecture $\ge$ `sm_70`) to compile without `#error` flags.

3. **Dual Execution Abstractions**:
   - **Standalone Binaries**: Detects `int main()` and executes the compiled binary directly.
   - **Library / Harness Export**: Detects `extern "C" void solve()` and dynamically invokes the compiled `.so` via Python `ctypes` FFI.

---

## Directory Structure

```
cuda-programs/
├── cuda/
│   ├── modal_run.py          # Serverless cloud GPU compilation & dispatch engine
│   ├── silu.cu               # Vectorized SiLU kernel with float4 alignment guards
│   ├── silu.cpp              # C++ translation unit variant for -x cu validation
│   ├── tiled_matmuls.cu      # Tiled matrix multiplication with shared memory SRAM
│   └── vector_add.cu         # 1M element vector addition baseline
├── cpp/
│   └── sum.cpp               # Reference CPU host array reduction
├── templates/                # Competitive programming & algorithmic reference templates
│   ├── template.cpp          # Fast I/O, PBDS order-statistics tree, local debug harness
│   ├── main.cpp              # Minimal problem skeleton
│   └── input.txt             # Local test input vector
├── .vscode/
│   ├── settings.json         # Clangd language server path and argument wiring
│   ├── tasks.json            # GCC compilation, local debug builds, and input pipes
│   └── c_cpp_properties.json # Fallback Microsoft C/C++ configurations
├── compile_commands.json     # Clangd compilation database mapping GCC libstdc++ headers
└── README.md                 # Technical research documentation
```

---

## Empirical Verification & Benchmark Runs

All experiments are verified through continuous cloud-based validation on NVIDIA Tesla T4 GPUs:

```bash
# Vectorized SiLU activation (Non-power-of-two 128x1027 matrix)
uv run modal run cuda/modal_run.py --file cuda/silu.cu
# Output:
# First 10 elements:
# Index      Input        SiLU(x)     
# 0          -5.000000    -0.033464   
# 1          -4.950000    -0.034816   
# 2          -4.900000    -0.036219   
# ...
# OK: SiLU verified for 128x1027 matrix

# Shared-memory tiled matrix multiplication (4x4 tile micro-benchmark)
uv run modal run cuda/modal_run.py --file cuda/tiled_matmuls.cu
# Output:
#     90   100   110   120
#    202   228   254   280
#    314   356   398   440
#    426   484   542   600

# 1M Element Vector Addition
uv run modal run cuda/modal_run.py --file cuda/vector_add.cu
# Output:
# OK: vector_add verified for 1048576 elements
```

---

## Local Development & C++ Toolchain

For competitive programming and algorithmic prototyping, the local environment is configured with Homebrew GCC:

- **Compiler**: `/opt/homebrew/bin/g++-16` (GCC 16.2.0, supporting `bits/stdc++.h` and Policy-Based Data Structures `__gnu_pbds`).
- **macOS SDK Root**: `SDKROOT` is exported in `~/.zshenv` to resolve system C library headers (`ctype.h`, `stdio.h`) across both interactive and non-interactive subshells.
- **AST IntelliSense**: Driven by LLVM `clangd` via `compile_commands.json` with explicit `-nostdinc++` and `-isystem` mappings resolving GCC `libstdc++` include hierarchies.

---

## Getting Started

### Prerequisites

- Python $\ge$ 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [Modal](https://modal.com) account

### Setup & Cloud Authentication

```bash
# Clone the repository
git clone https://github.com/DARK-art108/cuda-experiments.git
cd cuda-experiments

# Initialize virtual environment and install dependencies
uv sync

# Authenticate with Modal (one-time setup)
uv run modal setup
```

### Running Kernels

```bash
# Execute any CUDA kernel on a remote T4 GPU:
uv run modal run cuda/modal_run.py --file cuda/tiled_matmuls.cu
uv run modal run cuda/modal_run.py --file cuda/silu.cu
uv run modal run cuda/modal_run.py --file cuda/vector_add.cu
```
