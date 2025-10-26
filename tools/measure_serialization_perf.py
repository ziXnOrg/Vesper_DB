#!/usr/bin/env python3
import os
import sys
import time
import json
import tempfile
import shutil
import argparse
from pathlib import Path

# This script creates a synthetic IVF-PQ index (via the C++ binary if present) or
# uses the library through a small CLI helper if available, then measures:
# - serialized file size for v1.0 vs v1.1 (optionally compressed)
# - load() wall time across repetitions
# - optional mmap mode load() time
#
# It is intended as a developer tool; CI can call it and parse JSON output.

try:
    import psutil  # optional, for peak RSS if available
except Exception:
    psutil = None


def human(n):
    for unit in ["B","KB","MB","GB"]:
        if n < 1024.0:
            return f"{n:,.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"


def measure_load(binary: Path, index_dir: Path, reps: int = 5, extra_env=None):
    # We call the existing micro-benchmark and control its index location via env.
    env = os.environ.copy()
    env["VESPER_BENCH_INDEX_DIR"] = str(index_dir)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    # Prefer dedicated load bench if present
    bench = binary
    if not bench.exists():
        # Try default build location
        guess = Path("build/Release/vesper_bench_ivfpq_load.exe") if os.name == "nt" else Path("build/vesper_bench_ivfpq_load")
        if guess.exists():
            bench = guess
        else:
            raise SystemExit("Load benchmark binary not found. Build target vesper_bench_ivfpq_load first.")

    # Run and sample peak RSS if psutil is available
    import subprocess
    t0 = time.perf_counter()
    p = subprocess.Popen([str(bench), f"--benchmark_repetitions={reps}"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_bytes = 0
    proc = None
    if psutil is not None:
        try:
            proc = psutil.Process(p.pid)
        except Exception:
            proc = None
    try:
        while True:
            ret = p.poll()
            if proc is not None:
                try:
                    mi = proc.memory_info()
                    # On Windows, peak_wset is preferred if present; otherwise use rss
                    cur = getattr(mi, 'peak_wset', None)
                    if cur is None:
                        cur = getattr(mi, 'rss', 0)
                    peak_bytes = max(peak_bytes, int(cur))
                except Exception:
                    pass
            if ret is not None:
                break
            time.sleep(0.02)
        out, err = p.communicate()
    finally:
        t1 = time.perf_counter()
    if p.returncode != 0:
        print(out)
        print(err, file=sys.stderr)
        raise SystemExit("Benchmark failed")
    return (t1 - t0), out, peak_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=None, help="Path to vesper_bench_ivfpq_load binary")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--zstd_level", type=int, default=1)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="vesper_ivfpq_perf_"))
    try:
        # Generate baseline (v1.0)
        base_dir = tmp / "v10"
        base_dir.mkdir(parents=True, exist_ok=True)
        os.environ.pop("VESPER_IVFPQ_SAVE_V11", None)
        # The load bench will generate synthetic index if not present
        t_base, out_base, mem_base = measure_load(args.bench or Path(""), base_dir, reps=args.reps, extra_env={"VESPER_IVFPQ_LOAD_STREAM_CENTROIDS":"1"})
        size_base = (base_dir / "ivfpq.bin").stat().st_size if (base_dir / "ivfpq.bin").exists() else 0

        # v1.1 uncompressed
        v11_dir = tmp / "v11_raw"
        v11_dir.mkdir(parents=True, exist_ok=True)
        t_v11, out_v11, mem_v11 = measure_load(args.bench or Path(""), v11_dir, reps=args.reps, extra_env={"VESPER_IVFPQ_SAVE_V11":"1", "VESPER_IVFPQ_ZSTD_LEVEL":"0"})
        size_v11 = (v11_dir / "ivfpq.bin").stat().st_size if (v11_dir / "ivfpq.bin").exists() else 0

        # v1.1 compressed
        v11c_dir = tmp / "v11_zstd"
        v11c_dir.mkdir(parents=True, exist_ok=True)
        t_v11c, out_v11c, mem_v11c = measure_load(args.bench or Path(""), v11c_dir, reps=args.reps, extra_env={"VESPER_IVFPQ_SAVE_V11":"1", "VESPER_IVFPQ_ZSTD_LEVEL":str(args.zstd_level)})
        size_v11c = (v11c_dir / "ivfpq.bin").stat().st_size if (v11c_dir / "ivfpq.bin").exists() else 0

        report = {
            "size_bytes": {"v1_0": size_base, "v1_1_raw": size_v11, "v1_1_zstd": size_v11c},
            "size_readable": {"v1_0": human(size_base), "v1_1_raw": human(size_v11), "v1_1_zstd": human(size_v11c)},
            "load_wall_seconds": {"v1_0": t_base, "v1_1_raw": t_v11, "v1_1_zstd": t_v11c},
            "peak_memory_bytes": {"v1_0": mem_base, "v1_1_raw": mem_v11, "v1_1_zstd": mem_v11c},
            "peak_memory_readable": {"v1_0": human(mem_base), "v1_1_raw": human(mem_v11), "v1_1_zstd": human(mem_v11c)},
            "bench_output_sample": {"v1_0": out_base.splitlines()[-5:], "v1_1_raw": out_v11.splitlines()[-5:], "v1_1_zstd": out_v11c.splitlines()[-5:]}
        }
        print(json.dumps(report, indent=2))
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass

if __name__ == "__main__":
    main()

