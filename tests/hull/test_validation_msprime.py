"""msprime validation harness — Rust msinv core vs msprime.sim_ancestry.

Spec: docs/superpowers/specs/2026-04-30-msprime-validation-extension-design.md
(supersedes 2026-04-29-msprime-validation-design.md).

Each test calls ``_run_validation(scenario)`` which spawns two
subprocesses (one per engine) running ``tests/hull/_msprime_bench_runner``.
Per-rep stats arrive as JSON on the child's stdout; peak RSS is read
from ``os.wait4`` rusage. Pass criteria:

- moment stats (``pi_branch``, ``n_trees``, ``mean_tmrca``, ``dxy_branch``):
  ``|Δ| <= 3 * sqrt(SE_a^2 + SE_b^2)``
- AFS bin stats (``afs_*``): Bonferroni-corrected two-sided z, family-
  wise α = 0.003 across all AFS bins in that scenario.

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
ALPHA_FAMILY = 0.003  # family-wise α for both moment and AFS families


def _run_one_engine(scenario_name, engine, n_reps):
    """Spawn one child runner; return (per_rep_stats, per_rep_seconds,
    peak_rss_kb). Uses Popen + os.wait4 for clean per-child rusage."""
    cmd = [
        sys.executable, "-m", "tests.hull._msprime_bench_runner",
        "--scenario", scenario_name, "--engine", engine,
        "--n-reps", str(n_reps), "--seed-base", "0",
    ]
    # Popen does not auto-wait; we'll do it manually with os.wait4
    # so the per-child rusage is attributable.
    #
    # Pipe-buffer note: stdout JSON is ~78 KB at N_REPS=200 (N6 scenario,
    # 11 stats x 200 reps), exceeding the 64 KB Linux pipe buffer.  This
    # works because proc.stdout.read() blocks and drains the pipe as the
    # child writes.  Stderr drain happens AFTER stdout, so the child
    # MUST stay quiet on stderr (else the child blocks writing to a full
    # stderr pipe while the parent blocks reading stdout = deadlock).
    # Verified: the runner module emits no stderr.  If a future scenario
    # adds stderr output, switch to a threaded drain or merge stderr
    # into stdout (stderr=subprocess.STDOUT).
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Read all stdout / stderr without calling .wait()
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
    peak_kb = rusage.ru_maxrss  # Linux: KB
    return payload["per_rep_stats"], payload["per_rep_seconds"], peak_kb


def _mean_se(values):
    n = len(values)
    if n < 2:
        raise ValueError("need >= 2 reps to compute SE")
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(n)


def _bonferroni_z(k_bins, alpha=ALPHA_FAMILY):
    """Two-sided z bound for `k_bins` AFS bins at family-wise alpha."""
    per_bin = alpha / k_bins
    return statistics.NormalDist().inv_cdf(1.0 - per_bin / 2.0)


def _agg_engine_vals(per_rep_stats):
    """List of per-rep stat dicts -> dict[stat_name, list[value]]."""
    out: dict[str, list[float]] = {}
    for rep_stats in per_rep_stats:
        for k, v in rep_stats.items():
            out.setdefault(k, []).append(v)
    return out


def _run_validation(scenario_name, n_reps=N_REPS):
    """Run both engines via subprocess; assert per-stat agreement."""
    msinv_stats, msinv_secs, msinv_peak = _run_one_engine(
        scenario_name, "msinv", n_reps)
    msprime_stats, msprime_secs, msprime_peak = _run_one_engine(
        scenario_name, "msprime", n_reps)

    a = _agg_engine_vals(msinv_stats)
    b = _agg_engine_vals(msprime_stats)
    keys = list(a.keys())
    assert set(keys) == set(b.keys()), (
        f"stat key mismatch: msinv={set(keys)} vs msprime={set(b.keys())}")

    afs_keys = [k for k in keys if k.startswith("afs_")]
    moment_keys = [k for k in keys if not k.startswith("afs_")]
    z_afs = _bonferroni_z(len(afs_keys)) if afs_keys else None

    failures = []
    lines = []
    for k in moment_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = 3.0 * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: msinv={m_a:.4g} ± {se_a:.3g}, "
                f"msprime={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, 3·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)
    for k in afs_keys:
        m_a, se_a = _mean_se(a[k])
        m_b, se_b = _mean_se(b[k])
        bound = z_afs * math.sqrt(se_a ** 2 + se_b ** 2)
        delta = abs(m_a - m_b)
        ok = delta <= bound
        line = (f"{k}: msinv={m_a:.4g} ± {se_a:.3g}, "
                f"msprime={m_b:.4g} ± {se_b:.3g}, "
                f"|Δ|={delta:.4g}, {z_afs:.2f}·SE={bound:.4g} "
                f"→ {'OK' if ok else 'FAIL'}")
        lines.append(line)
        if not ok:
            failures.append(line)

    print(f"\n[{scenario_name}]")
    for line in lines:
        print(f"  {line}")

    _print_benchmark_block(
        scenario_name, msinv_secs, msprime_secs, msinv_peak, msprime_peak)

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


def _print_benchmark_block(scenario_name, msinv_secs, msprime_secs,
                           msinv_peak_kb, msprime_peak_kb):
    """Print + persist a per-scenario benchmark line."""
    m_mean, m_se = _mean_se(msinv_secs)
    p_mean, p_se = _mean_se(msprime_secs)
    m_total = sum(msinv_secs)
    p_total = sum(msprime_secs)
    m_rss_mb = msinv_peak_kb / 1024.0
    p_rss_mb = msprime_peak_kb / 1024.0
    print(f"[{scenario_name}] benchmarks")
    print(f"  msinv:   per-rep {m_mean*1000:6.1f} ms ± {m_se*1000:.1f}, "
          f"total {m_total:5.1f} s, peak RSS {m_rss_mb:6.1f} MB")
    print(f"  msprime: per-rep {p_mean*1000:6.1f} ms ± {p_se*1000:.1f}, "
          f"total {p_total:5.1f} s, peak RSS {p_rss_mb:6.1f} MB")
    print(f"  ratio:   per-rep msinv/msprime = {m_mean/p_mean:.2f}x;  "
          f"RAM msinv/msprime = {m_rss_mb/p_rss_mb:.2f}x")

    log_dir = pathlib.Path(".tmp")
    log_dir.mkdir(exist_ok=True)
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "scenario": scenario_name,
        "msinv": {"per_rep_s": m_mean, "per_rep_se": m_se,
                  "total_s": m_total, "peak_rss_mb": m_rss_mb},
        "msprime": {"per_rep_s": p_mean, "per_rep_se": p_se,
                    "total_s": p_total, "peak_rss_mb": p_rss_mb},
    }
    with (log_dir / "msprime_validation_bench.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def test_msprime_validation_n1_panmictic():
    """Rust msinv vs msprime — single-pop panmictic, n=10, ρ=40."""
    _run_validation("n1")


def test_msprime_validation_n2_two_pop_migration():
    """Rust msinv vs msprime — two-pop symmetric migration, M=1e-4."""
    _run_validation("n2")


def test_msprime_validation_n3_two_pop_split():
    """Rust msinv vs msprime — two-pop merge backward at T=2000."""
    _run_validation("n3")


def test_msprime_validation_n4_bottleneck():
    """Rust msinv vs msprime — Ne=10000 → 1000 (1000–2000 gens) → 10000."""
    _run_validation("n4")


def test_msprime_validation_n5_exponential_growth():
    """Rust msinv vs msprime — exponential growth, α=0.0005/gen."""
    _run_validation("n5")


def test_msprime_validation_n6_three_pop_with_split():
    """Rust msinv vs msprime — 3 pops, sym M=5e-5, merge to A at T=3000."""
    _run_validation("n6")
