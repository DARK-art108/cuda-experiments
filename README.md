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

## Local C++ (competitive programming)

The local toolchain is Homebrew GCC, not Apple clang — clang has neither `bits/stdc++.h` nor PBDS.

| Piece | Value |
|-------|-------|
| Compiler | `/opt/homebrew/bin/g++-16` (Homebrew GCC 16.2.0) |
| Aliases | `g++` → `g++-16`, `gcc` → `gcc-16` (`~/.zshrc`) |
| SDK sysroot | `SDKROOT` exported in `~/.zshenv` (GCC needs it to find macOS C headers) |
| Starter file | `templates/template.cpp` (fast IO, `-DLOCAL` debug printer, PBDS order-statistics tree) |
| Input file | `templates/input.txt` — used by `-DLOCAL` builds |

Compile and run by hand:

```bash
g++ -std=c++20 -O2 templates/template.cpp -o sol     # submission-clean build
g++ -std=c++20 -DLOCAL templates/template.cpp -o sol # debug build, reads input.txt
./sol
```

### Editor IntelliSense

`compile_commands.json` at the repo root drives both editors. It points clangd at GCC's
libstdc++ headers plus the macOS SDK, so `bits/stdc++.h` and PBDS resolve correctly.

- **VS Code** — extension `llvm-vs-code-extensions.vscode-clangd` + `.vscode/settings.json`
  (`clangd.path`). Build/run/debug via `.vscode/tasks.json`: `build`, `run`,
  `debug build (LOCAL)`, `run with input.txt (LOCAL)`.
- **Sublime Text** — LSP + LSP-clangd, configured in
  `Packages/User/LSP-clangd.sublime-settings` (`system_binary` = Homebrew clangd).

After adding, renaming or moving a `.cpp` file, regenerate the database:

```bash
python3 - <<'EOF'
import json, os, subprocess
root, sdk = os.getcwd(), subprocess.check_output(['xcrun','--show-sdk-path'], text=True).strip()
gcc, gxx = '/opt/homebrew/Cellar/gcc/16.2.0', '/opt/homebrew/Cellar/gcc/16.2.0/lib/gcc/current/gcc/aarch64-apple-darwin24/16'
files = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ('.venv', '__pycache__', '.git', 'build')]
    for f in fn:
        if f.endswith('.cpp'):
            src = open(os.path.join(dp, f), errors='ignore').read()
            if 'cuda_runtime.h' not in src:
                files.append(os.path.relpath(os.path.join(dp, f), root))
files.sort()
args = ['/opt/homebrew/opt/llvm/bin/clang++','-std=gnu++20','-w','-nostdinc++',
        '-D__GLIBCXX_TYPE_INT_N_0=__int128','-D__GLIBCXX_BITSIZE_INT_N_0=128','-D__aarch64__=0',
        '-isystem',f'{gcc}/include/c++/16','-isystem',f'{gcc}/include/c++/16/aarch64-apple-darwin24',
        '-isystem',f'{gcc}/include/c++/16/backward','-isystem',f'{gxx}/include',
        '-isystem',f'{gxx}/include-fixed','-isystem',f'{sdk}/usr/include']
json.dump([{"directory": root, "arguments": args + [f], "file": f} for f in files],
          open(os.path.join(root, 'compile_commands.json'), 'w'), indent=1)
EOF
```

CUDA sources (`cuda/*.cu`, `cuda/silu.cpp`) are deliberately excluded — they need the CUDA
toolkit headers and are meant to be compiled on Modal, not locally.

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
