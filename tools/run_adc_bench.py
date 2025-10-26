#!/usr/bin/env python3
"""
One-shot runner for the IVF-PQ ADC microbenchmark that reports AVX512/AVX2 speedup.

It will:
  1) Build the vesper_bench_ivfpq_adc target (by default)
  2) Run it with Google Benchmark JSON output
  3) Parse results and print a small summary table with speedups

Usage examples:
  python tools/run_adc_bench.py
  python tools/run_adc_bench.py --no-build
  python tools/run_adc_bench.py --build-dir build --config Release --min-time 1.5

Environment override for SIMD selection (forwarded by the benchmark itself):
  VESPER_AVX512=0   # force disable AVX-512
  VESPER_AVX512=1   # force enable AVX-512 (compile-time guards still apply)

Note: If AVX-512 is not supported at runtime, the AVX512 entries will be absent
and the script will skip speedup computation for those rows.
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Dict, Tuple
import math


def run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def build_target(build_dir: str, config: str, target: str) -> None:
    # MSVC uses multi-config builds; others typically use single-config
    if platform.system() == "Windows":
        cmd = ["cmake", "--build", build_dir, "--config", config, "--target", target, "-j", "8"]
    else:
        cmd = ["cmake", "--build", build_dir, "--target", target, "-j", "8"]
    res = run(cmd)
    if res.returncode != 0:
        print(res.stdout)
        raise SystemExit(f"Build failed for target {target}")


def exe_path(build_dir: str, config: str, name: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(build_dir, config, name + ".exe")
    else:
        return os.path.join(build_dir, name)


def unit_to_seconds(value: float, unit: str) -> float:
    if unit == "s":
        return value
    if unit == "ms":
        return value / 1e3
    if unit == "us":
        return value / 1e6
    if unit == "ns":
        return value / 1e9
    # Unknown: assume seconds
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADC microbench and print AVX512/AVX2 speedup")
    parser.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    parser.add_argument("--config", default=("Release" if platform.system()=="Windows" else ""), help="CMake build config on multi-config generators (Windows: Release/Debug)")
    parser.add_argument("--no-build", action="store_true", help="Do not build the target before running")
    parser.add_argument("--min-time", type=float, default=1.0, help="Google Benchmark --benchmark_min_time in seconds")
    args = parser.parse_args()

    target = "vesper_bench_ivfpq_adc"

    if not args.no_build:
        build_target(args.build_dir, args.config, target)

    exe = exe_path(args.build_dir, args.config, target)
    if not os.path.exists(exe):
        raise SystemExit(f"Benchmark executable not found: {exe}. Try --build-dir/--config or run without --no-build.")

    out_json = os.path.join(args.build_dir, "adc_bench.json")
    bench_cmd = [exe, f"--benchmark_min_time={args.min_time}s", f"--benchmark_out={out_json}", "--benchmark_out_format=json"]
    res = run(bench_cmd)
    if res.returncode != 0:
        print(res.stdout)
        raise SystemExit("Benchmark run failed")

    try:
        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(res.stdout)
        raise SystemExit(f"Failed to read/parse JSON output: {e}")

    # Collect results per (dim,m,nprobe) triple
    # Names look like: BenchADC_AVX512/64/8/4
    rows: Dict[Tuple[int,int,int], Dict[str, float]] = {}

    for b in payload.get("benchmarks", []):
        name = b.get("name") or b.get("run_name") or ""
        if "BenchADC_" not in name:
            continue
        parts = name.split("/")
        # Expect parts[0] like BenchADC_AVX2, then /dim/m[/nprobe]
        if len(parts) < 3:
            continue
        impl = parts[0].removeprefix("BenchADC_")
        try:
            dim = int(parts[1])
            m = int(parts[2])
            nprobe = int(parts[3]) if len(parts) >= 4 else 1
        except ValueError:
            continue

        # Prefer real_time; fallback to cpu_time
        time_val = b.get("real_time", b.get("cpu_time"))
        time_unit = b.get("time_unit", "ns")
        if time_val is None:
            continue
        sec = unit_to_seconds(float(time_val), time_unit)

        key = (dim, m, nprobe)
        rows.setdefault(key, {})[impl.lower()] = sec

    # Print summary
    print("\nADC microbenchmark summary (lower is better):")
    print("{:<6} {:<5} {:<7} {:>12} {:>12} {:>10} {:>10}".format("dim", "m", "nprobe", "scalar(s)", "avx2(s)", "avx512(s)", "512/2x"))
    for (dim, m, nprobe), vals in sorted(rows.items()):
        scalar = vals.get("scalar")
        avx2 = vals.get("AVX2".lower())
        avx512 = vals.get("AVX512".lower())

        spd = (avx2 / avx512) if (avx2 and avx512 and avx512 > 0) else None
        print("{:<6} {:<5} {:<7} {:>12} {:>12} {:>10} {:>10}".format(
            dim,
            m,
            nprobe,
            f"{scalar:.6f}" if scalar else "-",
            f"{avx2:.6f}" if avx2 else "-",
            f"{avx512:.6f}" if avx512 else "-",
            f"{spd:.2f}x" if spd else "-",
        ))

    # Aggregate geometric mean speedup across all comparable cases
    logs = []
    for vals in rows.values():
        avx2 = vals.get("avx2")
        avx512 = vals.get("avx512")
        if avx2 and avx512 and avx512 > 0:
            logs.append(math.log(avx2 / avx512))
    if logs:
        gm = math.exp(sum(logs) / len(logs))
        print(f"\nAggregate geometric mean speedup (AVX-512 vs AVX2): {gm:.2f}x across {len(logs)} cases")
    else:
        print("\nAggregate geometric mean speedup: n/a (no AVX-512 results present)")

    print("\nNotes:")
    print("- Use VESPER_AVX512=0/1 to force disable/enable AVX-512, or leave unset to auto-detect.")
    cpp = os.getenv("VESPER_ADC_CODES_PER_PROBE", "2048")
    print(f"- Codes per probe can be adjusted via VESPER_ADC_CODES_PER_PROBE (current: {cpp}).")
    print("- The speedup column is AVX-512 time divided into AVX2 time (>=1.3x target).")


if __name__ == "__main__":
    main()

