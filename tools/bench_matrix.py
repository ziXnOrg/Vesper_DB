#!/usr/bin/env python3
import os
import subprocess
import sys
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, 'build')

@dataclass
class RunResult:
    ok: bool
    rate: Optional[float]
    coverage: Optional[float]
    note: str = ''


def run_cmd(cmd, env=None, cwd=ROOT):
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode, proc.stdout


def configure_build(serialize: bool, accelerate: bool) -> bool:
    args = [
        'cmake', '-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release',
        f'-DVESPER_SERIALIZE_BASE_LAYER={"ON" if serialize else "OFF"}',
        f'-DVESPER_ENABLE_ACCELERATE={"ON" if accelerate else "OFF"}',
    ]
    rc, _ = run_cmd(' '.join(args))
    return rc == 0


def build_targets() -> bool:
    rc, _ = run_cmd('cmake --build build --target test_hnsw_batch hnsw_connectivity_test -j')
    return rc == 0


def parse_small(stdout: str) -> RunResult:
    # Expect lines like: "Added in X seconds (Y vec/sec)" and Coverage: Z%
    rate = None
    cov = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if 'vec/sec' in line and 'Added in' in line and '(' in line:
            try:
                part = line.split('(')[1]
                rate = float(part.split('vec/sec')[0].strip())
            except Exception:
                pass
        if 'Coverage:' in line and '%' in line:
            try:
                cov_str = line.split('Coverage:')[1].split('%')[0]
                cov = float(cov_str.strip())
            except Exception:
                pass
    ok = (rate is not None and cov is not None and cov >= 95.0)
    return RunResult(ok, rate, cov)


def parse_large(stdout: str) -> RunResult:
    # Expect lines like: Rate: Y vec/sec and Coverage: Z%
    rate = None
    cov = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if 'Rate:' in line and 'vec/sec' in line:
            try:
                rate = float(line.split('Rate:')[1].split('vec/sec')[0].strip())
            except Exception:
                pass
        if 'Coverage:' in line and '%' in line:
            try:
                cov_str = line.split('Coverage:')[1].split('%')[0]
                cov = float(cov_str.strip())
            except Exception:
                pass
    ok = (rate is not None and cov is not None and cov >= 95.0)
    return RunResult(ok, rate, cov)


def run_exec(path: str, env: dict) -> tuple[bool, str]:
    rc, out = run_cmd(shlex.quote(path), env=env)
    return rc == 0, out


def main():
    combos = []
    # Matrix focused on requested points: serialize OFF/ON, accelerate OFF/ON, threads 2,3,4,6, adaptive off/on
    for serialize in [False, True]:
        for accelerate in [False, True]:
            combos.append((serialize, accelerate))

    thread_vals = [2, 3, 4, 6]
    adaptive_vals = [False, True]

    summary = []

    for serialize, accelerate in combos:
        if not configure_build(serialize, accelerate):
            print('Configure failed, skipping this combo')
            continue
        if not build_targets():
            print('Build failed, skipping this combo')
            continue

        for threads in thread_vals:
            for adaptive in adaptive_vals:
                env = os.environ.copy()
                env['VESPER_NUM_THREADS'] = str(threads)
                env['VESPER_EFC'] = '150'
                if adaptive:
                    env['VESPER_ADAPTIVE_EF'] = '1'
                else:
                    env.pop('VESPER_ADAPTIVE_EF', None)
                # Leave VESPER_EFC_UPPER unset for auto

                ok1, out1 = run_exec(os.path.join(BUILD_DIR, 'test_hnsw_batch'), env)
                res1 = parse_small(out1) if ok1 else RunResult(False, None, None, note='exec_failed')

                ok2, out2 = run_exec(os.path.join(BUILD_DIR, 'hnsw_connectivity_test'), env)
                res2 = parse_large(out2) if ok2 else RunResult(False, None, None, note='exec_failed')

                summary.append({
                    'serialize': 'ON' if serialize else 'OFF',
                    'accelerate': 'ON' if accelerate else 'OFF',
                    'threads': threads,
                    'adaptive': adaptive,
                    'small_rate': res1.rate,
                    'small_cov': res1.coverage,
                    'large_rate': res2.rate,
                    'large_cov': res2.coverage,
                    'ok': res1.ok and res2.ok,
                    'note': res1.note or res2.note,
                })

    # Print summary table
    print('\n=== HNSW Build Matrix Summary ===')
    print('serialize accel th adapt | small_rate  small_cov | large_rate  large_cov | ok')
    for row in summary:
        small_rate = 'NA' if row['small_rate'] is None else f"{row['small_rate']:.0f}"
        small_cov = 'NA' if row['small_cov'] is None else f"{row['small_cov']:.2f}%"
        large_rate = 'NA' if row['large_rate'] is None else f"{row['large_rate']:.0f}"
        large_cov = 'NA' if row['large_cov'] is None else f"{row['large_cov']:.2f}%"
        print(f"{row['serialize']:>8} {row['accelerate']:>5} {row['threads']:>2} {str(row['adaptive']):>5} | "
              f"{small_rate:>10}  {small_cov:>9} | {large_rate:>10}  {large_cov:>9} | "
              f"{'PASS' if row['ok'] else 'FAIL'}")

    # Exit code is 0 if all ok
    all_ok = all(r['ok'] for r in summary) if summary else False
    sys.exit(0 if all_ok else 1)

if __name__ == '__main__':
    main()

