"""
Generic Modal GPU Runner
========================
Supports .cu (CUDA) and .cpp files.

The compiler is chosen by *content*, not extension: any source that looks like CUDA
(cuda_runtime.h, __global__, <<<, cooperative_groups, cuda/pipeline, ...) is built with
nvcc — a .cpp file gets an explicit `-x cu` — and everything else is built with g++.

Convention — pick one:
  1. Write a full program with `int main()` → compiled & run as executable
  2. Write `extern "C" void solve() { ... }` (no params) → compiled as .so, solve() is called

Usage:
  modal run cuda/modal_run.py --file path/to/program.cu
  modal run cuda/modal_run.py --file path/to/program.cpp
"""

import modal
import os
import sys

app = modal.App("gpu-runner")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
)

# Tokens that mean "this translation unit is CUDA, whatever the file extension says".
CUDA_MARKERS = (
    "cuda_runtime.h", "cuda/pipeline", "cooperative_groups",
    "cudaMalloc", "cudaMemcpy", "cudaFree", "cudaDeviceSynchronize",
    "__global__", "__device__", "__shared__", "<<<",
)

OUT_BIN = "/root/program"
OUT_SO = "/root/program.so"


def is_cuda_source(source: str, filename: str) -> bool:
    """CUDA code can live in a .cpp file — decide by content, not extension."""
    if os.path.splitext(filename)[1].lower() == ".cu":
        return True
    return any(marker in source for marker in CUDA_MARKERS)


def build_commands(source: str, filename: str):
    """Return (exe_cmd, so_cmd) for the given source.

    nvcc picks the language from the file extension, so non-.cu CUDA files
    need an explicit `-x cu`. Flags must precede the input file.
    """
    src_path = f"/root/{filename}"

    if is_cuda_source(source, filename):
        force_cu = [] if filename.endswith(".cu") else ["-x", "cu"]
        arch = ["-arch=native"]  # target the GPU actually attached to this container
        return (
            ["nvcc", "-std=c++17", "-O2", *arch, *force_cu, "-o", OUT_BIN, src_path],
            ["nvcc", "-std=c++17", "-O2", *arch, *force_cu, "-shared", "-Xcompiler", "-fPIC",
             "-o", OUT_SO, src_path],
        )

    return (
        ["g++", "-std=c++20", "-O2", "-o", OUT_BIN, src_path],
        ["g++", "-std=c++20", "-O2", "-shared", "-fPIC", "-o", OUT_SO, src_path],
    )


# ── Remote function ────────────────────────────────────────────────────────────

@app.function(gpu="T4", image=cuda_image)
def run_source(source: str, filename: str) -> str:
    import subprocess
    import ctypes
    import tempfile
    import os

    src_path = f"/root/{filename}"

    # Write source to container
    with open(src_path, "w") as f:
        f.write(source)

    has_main = "int main(" in source or "int main (" in source
    has_solve = "void solve()" in source  # parameterless solve

    # ── Compiler selection (nvcc for CUDA, g++ for plain C++) ────────────────
    exe_cmd, so_cmd = build_commands(source, filename)

    # ── Path 1: has main → build & run executable ─────────────────────────────
    if has_main:
        res = subprocess.run(exe_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return f"[COMPILE ERROR]\n{res.stderr.strip()}"
        out = subprocess.run([OUT_BIN], capture_output=True, text=True)
        combined = out.stdout + out.stderr
        if out.returncode != 0:
            return f"[RUNTIME ERROR (exit {out.returncode})]\n{combined.strip()}"
        return combined.strip()

    # ── Path 2: parameterless solve() → build .so and call it ─────────────────
    if has_solve:
        res = subprocess.run(so_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return f"[COMPILE ERROR]\n{res.stderr.strip()}"
        lib = ctypes.CDLL(OUT_SO)
        lib.solve.restype  = None
        lib.solve.argtypes = []
        lib.solve()
        return "[solve() completed — check logs for any printf/cout output]"

    return (
        "[ERROR] File must contain either:\n"
        "  • int main()           — for a standalone executable\n"
        "  • extern \"C\" void solve()  — for a library-style entry point"
    )


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(file: str):
    """Run a .cu or .cpp file on a Modal T4 GPU.

    Examples
    --------
    modal run modal_run.py --file vector_add.cu
    modal run modal_run.py --file ../cpp/sum.cpp
    """
    path = os.path.abspath(file)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".cu", ".cpp", ".cc", ".cxx"):
        print(f"[ERROR] Unsupported extension '{ext}'. Use .cu or .cpp", file=sys.stderr)
        raise SystemExit(1)

    print(f"→ Uploading {filename} ...")
    with open(path) as f:
        source = f.read()

    print(f"→ Running on Modal T4 GPU ...\n")
    result = run_source.remote(source, filename)
    print(result)
