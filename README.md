# cuda-experiments 🚀

A collection of CUDA and C++ GPU programming experiments, running on cloud GPUs via [Modal](https://modal.com).

## Structure

```
cuda-experiments/
├── modal_run.py          # Generic Modal runner (run any .cu or .cpp)
├── vector_add.cu         # Vector addition — standalone (has main)
├── cuda/
│   └── vector_add.cu     # Vector addition — solve() style
└── cpp/
    └── sum.cpp           # C++ array sum example
```

## Running on Modal GPU

Any `.cu` or `.cpp` file can be run on a cloud T4 GPU with a single command:

```bash
modal run modal_run.py --file <path-to-file>
```

### Examples

```bash
modal run modal_run.py --file vector_add.cu
modal run modal_run.py --file cuda/vector_add.cu
modal run modal_run.py --file cpp/sum.cpp
```

## Writing Compatible Files

The runner supports two conventions:

### 1. Standalone — `int main()`
Write a complete program. Works for both `.cu` and `.cpp`.

```cpp
// example.cu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void my_kernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= 2.0f;
}

int main() {
    // allocate, launch kernel, verify, print
}
```

### 2. Solve style — `extern "C" void solve()`
Parameterless `solve()` — handle everything inside (alloc, compute, print, free).

```cpp
// example.cu
#include <cuda_runtime.h>

__global__ void my_kernel(...) { ... }

extern "C" void solve() {
    // full self-contained logic
}
```

## Setup

```bash
pip install modal
modal setup     # authenticate once
```

## Requirements

- Python ≥ 3.12
- [Modal](https://modal.com) account (free tier available)

## Experiments

| File | Description |
|------|-------------|
| `vector_add.cu` | Element-wise addition of two float arrays on GPU |
| `cuda/vector_add.cu` | Same, in competitive-style `solve()` format |
| `cpp/sum.cpp` | Array sum in standard C++ |
