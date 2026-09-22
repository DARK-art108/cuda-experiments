"""
Generic Modal GPU Runner
========================
Supports .cu (CUDA) and .cpp files.

Convention — pick one:
  1. Write a full program with `int main()` → compiled & run as executable
  2. Write `extern "C" void solve() { ... }` (no params) → compiled as .so, solve() is called

Usage:
  modal run modal_run.py --file path/to/program.cu
  modal run modal_run.py --file path/to/program.cpp
"""

import modal
import os
import sys

app = modal.App("gpu-runner")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
)


# ── Remote function ────────────────────────────────────────────────────────────

@app.function(gpu="T4", image=cuda_image)
def run_source(source: str, filename: str) -> str:
    import subprocess
    import ctypes
    import tempfile
    import os

    ext = os.path.splitext(filename)[1].lower()   # ".cu" or ".cpp"
    src_path = f"/root/{filename}"
    out_bin  = "/root/program"
    out_so   = "/root/program.so"

    # Write source to container
    with open(src_path, "w") as f:
        f.write(source)

    has_main  = "int main(" in source or "int main (" in source
    has_solve = "void solve()" in source  # parameterless solve

    # ── Compiler selection ────────────────────────────────────────────────────
    def nvcc(*flags):
        return ["nvcc", "-std=c++17", "-O2", *flags, src_path]

    def gpp(*flags):
        return ["g++", "-std=c++20", "-O2", *flags, src_path]

    if ext == ".cu":
        exe_cmd = nvcc("-o", out_bin)
        so_cmd  = nvcc("-shared", "-Xcompiler", "-fPIC", "-o", out_so)
    else:
        exe_cmd = gpp("-o", out_bin)
        so_cmd  = gpp("-shared", "-fPIC", "-o", out_so)

    # ── Path 1: has main → build & run executable ─────────────────────────────
    if has_main:
        res = subprocess.run(exe_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return f"[COMPILE ERROR]\n{res.stderr.strip()}"
        out = subprocess.run([out_bin], capture_output=True, text=True)
        combined = out.stdout + out.stderr
        if out.returncode != 0:
            return f"[RUNTIME ERROR (exit {out.returncode})]\n{combined.strip()}"
        return combined.strip()

    # ── Path 2: parameterless solve() → build .so and call it ─────────────────
    if has_solve:
        res = subprocess.run(so_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return f"[COMPILE ERROR]\n{res.stderr.strip()}"
        lib = ctypes.CDLL(out_so)
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
