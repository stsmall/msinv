"""Shared helpers for cross-simulator validation harnesses.

Used by both the msprime track (`test_validation_msprime.py`) and the
discoal track (`test_validation_discoal.py`).  The parent-side flow
(spawn child runners per engine, read JSON stats, apply pass criteria,
print + persist benchmark) is identical across tracks; only the child
runner module path differs.

Platform note: peak RSS uses ``rusage.ru_maxrss``, which is **kilobytes
on Linux** but **bytes on macOS/BSD**.  This harness assumes Linux; on
other platforms the reported peak RSS will be off by 1024×.  Comparison
stats (the actual pass/fail criteria) are platform-independent.
"""

import datetime
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys


N_REPS = 200
ALPHA_FAMILY = 0.003  # family-wise α for both moment and AFS/window families


def _run_one_engine(runner_module, scenario_name, engine, n_reps):
    """Spawn one child runner; return (per_rep_stats, per_rep_seconds,
    peak_rss_kb).

    `runner_module` is the dotted path passed to ``python -m``, e.g.
    ``"tests.hull._msprime_bench_runner"`` or
    ``"tests.hull._discoal_bench_runner"``.

    Uses Popen + os.wait4 for clean per-child rusage.

    Pipe-buffer note: stdout JSON is ~78 KB at N_REPS=200 (msprime
    track N6, 11 stats x 200 reps), exceeding the 64 KB Linux pipe
    buffer.  This works because proc.stdout.read() blocks and drains
    the pipe as the child writes.  Stderr drain happens AFTER stdout,
    so the child MUST stay quiet on stderr (else the child blocks
    writing to a full stderr pipe while the parent blocks reading
    stdout = deadlock).  Both runners must redirect any third-party
    stdout/stderr (e.g. discoal's DEBUG output) to /dev/null.
    """
    cmd = [
        sys.executable, "-m", runner_module,
        "--scenario", scenario_name, "--engine", engine,
        "--n-reps", str(n_reps), "--seed-base", "0",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = proc.stdout.read()
    stderr = proc.stderr.read()
    proc.stdout.close()
    proc.stderr.close()
    pid, status, rusage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner failed for {scenario_name}/{engine} "
            f"(rc={proc.returncode}):\n{stderr.decode()}")
    payload = json.loads(stdout.decode())
    peak_kb = rusage.ru_maxrss
    return payload["per_rep_stats"], payload["per_rep_seconds"], peak_kb


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _bonferroni_z(k_bins, alpha=ALPHA_FAMILY):
    """Two-sided z bound for `k_bins` AFS or window bins at family-wise alpha."""
    per_bin = alpha / k_bins
    return statistics.NormalDist().inv_cdf(1.0 - per_bin / 2.0)


def _agg_engine_vals(per_rep_stats):
    """List of per-rep stat dicts -> dict[stat_name, list[value]]."""
    out: dict[str, list[float]] = {}
    for rep_stats in per_rep_stats:
        for k, v in rep_stats.items():
            out.setdefault(k, []).append(v)
    return out


def _run_validation(runner_module, scenario_name, engine_a, engine_b,
                    bench_log, n_reps=N_REPS):
    """Run two engines via subprocess; assert per-stat agreement.

    `engine_a` and `engine_b` are engine names passed to the runner
    (e.g. "msinv", "msprime", "discoal").  `bench_log` is the path
    (str or Path) to the JSONL benchmark log to append.
    """
    a_stats, a_secs, a_peak = _run_one_engine(
        runner_module, scenario_name, engine_a, n_reps)
    b_stats, b_secs, b_peak = _run_one_engine(
        runner_module, scenario_name, engine_b, n_reps)

    a = _agg_engine_vals(a_stats)
    b = _agg_engine_vals(b_stats)
    keys = list(a.keys())
    assert set(keys) == set(b.keys()), (
        f"stat key mismatch: {engine_a}={set(keys)} vs "
        f"{engine_b}={set(b.keys())}")

    afs_keys = [k for k in keys if k.startswith("afs_")]
    win_keys = [k for k in keys if k.startswith("pi_window_")]
    moment_keys = [
        k for k in keys
        if not k.startswith("afs_") and not k.startswith("pi_window_")]
    z_afs = _bonferroni_z(len(afs_keys)) if afs_keys else None
    z_win = _bonferroni_z(len(win_keys)) if win_keys else None

    failures = []
    lines = []
    for k in moment_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = 3.0 * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: {engine_a}={m_a:.4g} ± {se_a:.3g}, "
                f"{engine_b}={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, 3·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)
    for family_keys, z_family, label in (
            (afs_keys, z_afs, f"{z_afs:.2f}·SE" if z_afs else None),
            (win_keys, z_win, f"{z_win:.2f}·SE" if z_win else None)):
        if not family_keys:
            continue
        for k in family_keys:
            m_a, se_a = _mean_se(a[k])
            m_b, se_b = _mean_se(b[k])
            bound = z_family * math.sqrt(se_a ** 2 + se_b ** 2)
            delta = abs(m_a - m_b)
            ok = delta <= bound
            line = (f"{k}: {engine_a}={m_a:.4g} ± {se_a:.3g}, "
                    f"{engine_b}={m_b:.4g} ± {se_b:.3g}, "
                    f"|Δ|={delta:.4g}, {label}={bound:.4g} "
                    f"→ {'OK' if ok else 'FAIL'}")
            lines.append(line)
            if not ok:
                failures.append(line)

    print(f"\n[{scenario_name}]")
    for line in lines:
        print(f"  {line}")

    _print_benchmark_block(
        scenario_name, engine_a, engine_b, a_secs, b_secs,
        a_peak, b_peak, bench_log)

    if failures:
        raise AssertionError(
            f"\n[{scenario_name}]\n  " + "\n  ".join(lines))


def _git_short_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def _print_benchmark_block(scenario_name, engine_a, engine_b,
                           a_secs, b_secs, a_peak_kb, b_peak_kb,
                           bench_log):
    """Print + persist a per-scenario benchmark line."""
    a_mean, a_se = _mean_se(a_secs)
    b_mean, b_se = _mean_se(b_secs)
    a_total = sum(a_secs)
    b_total = sum(b_secs)
    a_rss_mb = a_peak_kb / 1024.0
    b_rss_mb = b_peak_kb / 1024.0
    print(f"[{scenario_name}] benchmarks")
    print(f"  {engine_a:7s}: per-rep {a_mean*1000:6.1f} ms ± {a_se*1000:.1f}, "
          f"total {a_total:5.1f} s, peak RSS {a_rss_mb:6.1f} MB")
    print(f"  {engine_b:7s}: per-rep {b_mean*1000:6.1f} ms ± {b_se*1000:.1f}, "
          f"total {b_total:5.1f} s, peak RSS {b_rss_mb:6.1f} MB")
    print(f"  ratio:   per-rep {engine_a}/{engine_b} = {a_mean/b_mean:.2f}x;  "
          f"RAM {engine_a}/{engine_b} = {a_rss_mb/b_rss_mb:.2f}x")

    log_dir = pathlib.Path(bench_log).parent
    log_dir.mkdir(exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "scenario": scenario_name,
        engine_a: {"per_rep_s": a_mean, "per_rep_se": a_se,
                   "total_s": a_total, "peak_rss_mb": a_rss_mb},
        engine_b: {"per_rep_s": b_mean, "per_rep_se": b_se,
                   "total_s": b_total, "peak_rss_mb": b_rss_mb},
    }
    with pathlib.Path(bench_log).open("a") as f:
        f.write(json.dumps(record) + "\n")
