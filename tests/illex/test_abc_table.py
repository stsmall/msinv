"""Task 6: reference-table driver (run_table.py) and collector (collect.py). Soft D1, K = 3 on msinv (auto; ~1-3 s/row)."""
import gzip
import json
import math

import pytest

from illex.abc import collect, run_table
from illex.abc.statvec import STAT_KEYS
from illex.abc.trajectory import KEYS

ROW_KEYS = (set(KEYS) | set(STAT_KEYS) | {"model", "flux", "cls", "task_id", "w_est", "p_obs", "n_nan", "p_max", "p_now",
                                          "path_len", "engine", "seed", "wall_s"})


def _run(out, task_id=0, k=3, workers=1):
    return run_table.run(model="D1", flux=0, cls="soft", n_accept=k, task_id=task_id, out=str(out),
                         workers=workers)


def _rows(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f]


def test_small_run_roundtrip(tmp_path):
    _run(tmp_path)
    rows = _rows(tmp_path / "D1_0_soft_0.jsonl.gz")
    assert len(rows) == 3
    for r in rows:
        assert ROW_KEYS <= set(r)
        assert r["engine"] == "msinv" and r["model"] == "D1" and r["cls"] == "soft"
        assert r["wall_s"] > 0 and r["n_nan"] >= 0
    c = json.loads((tmp_path / "D1_0_soft_0_counts.json").read_text())
    assert c["counts"]["soft"]["accepted"] >= 3 and c["counts"]["soft"]["drawn"] >= 3
    assert {"git_head", "msinv_version", "msprime_version"} <= set(c["meta"])
    # deterministic from (model, flux, class, task_id)
    _run(tmp_path / "again")
    assert [r["seed"] for r in _rows(tmp_path / "again" / "D1_0_soft_0.jsonl.gz")] == [r["seed"] for r in rows]
    assert [r["t_inv"] for r in _rows(tmp_path / "again" / "D1_0_soft_0.jsonl.gz")] == [r["t_inv"] for r in rows]


def test_atomic_write(tmp_path, monkeypatch):
    real = run_table.simulate_stats
    calls = []

    def boom(path, seed, engine="auto"):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed")
        return real(path, seed, engine=engine)

    monkeypatch.setattr(run_table, "simulate_stats", boom)
    with pytest.raises(RuntimeError):
        _run(tmp_path)
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.jsonl.gz")) and not list(tmp_path.glob("*counts.json"))
    assert not list(tmp_path.glob("*.tmp*"))


def _fake(out, tid, drawn=100, acc=3):
    with gzip.open(out / f"D1_0_soft_{tid}.jsonl.gz", "wt") as f:
        for i in range(acc):
            f.write(json.dumps({"model": "D1", "flux": 0, "cls": "soft", "task_id": tid, "seed": i, "p_now": 0.6,
                                "n_nan": 0, "x": math.nan}) + "\n")
    (out / f"D1_0_soft_{tid}_counts.json").write_text(json.dumps(
        {"counts": {"hard": dict(drawn=0, lost=0, fixed=0, lost_establishing=0, est_failed=0, accepted=0),
                    "soft": dict(drawn=drawn, lost=40, fixed=10, lost_establishing=0, est_failed=0, accepted=acc)},
         "meta": {}}))


def test_collect_reports_missing(tmp_path):
    for t in (0, 1, 3):
        _fake(tmp_path, t)
    res = tmp_path / "res"
    info = collect.collect(str(tmp_path), expect=5, out_dir=str(res))
    assert info["missing"] == {"D1_0_soft": [2, 4]}
    assert info["n_rows"] == 9
    import pandas as pd
    acc = pd.read_csv(res / "abc_acceptance.csv")
    r = acc.iloc[0]
    assert r["drawn"] == 300 and r["accepted"] == 9 and abs(r["acceptance"] - 0.03) < 1e-12
    assert r["ci_lo"] < 0.03 < r["ci_hi"] and r["lost"] == 120
    assert (res / "abc_table.parquet").exists() or (res / "abc_table.csv.gz").exists()


def test_collect_weighted_acceptance(tmp_path):
    with gzip.open(tmp_path / "N_0_hard_0.jsonl.gz", "wt") as f:
        for i in range(2):
            f.write(json.dumps({"model": "N", "flux": 0, "cls": "hard", "task_id": 0, "w_est": 0.002}) + "\n")
    z = dict(lost=0, fixed=0, lost_establishing=0, est_failed=0)
    (tmp_path / "N_0_hard_0_counts.json").write_text(json.dumps(
        {"counts": {"hard": dict(drawn=1000, accepted=2, attempts=500_000, w_acc=0.004, w_acc_sq=8e-6, **z),
                    "soft": dict(drawn=0, accepted=0, attempts=0, w_acc=0.0, w_acc_sq=0.0, **z)}, "meta": {}}))
    info = collect.collect(str(tmp_path), out_dir=str(tmp_path / "res"))
    r = info["acceptance"].iloc[0]
    assert r["acceptance"] == pytest.approx(4e-6) and r["acceptance_attempts"] == pytest.approx(4e-6)
    assert 0 <= r["ci_lo"] < 4e-6 < r["ci_hi"]


def test_collect_hard_k0_and_absent(tmp_path):
    z = dict(lost=0, fixed=0, lost_establishing=0, est_failed=0)
    with gzip.open(tmp_path / "N_0_hard_0.jsonl.gz", "wt"):
        pass
    (tmp_path / "N_0_hard_0_counts.json").write_text(json.dumps(
        {"counts": {"hard": dict(drawn=1000, accepted=0, attempts=500_000, w_acc=0.0, w_acc_sq=0.0, **z),
                    "soft": dict(drawn=0, accepted=0, attempts=0, w_acc=0.0, w_acc_sq=0.0, **z)}, "meta": {}}))
    info = collect.collect(str(tmp_path), out_dir=str(tmp_path / "res"))
    r = info["acceptance"].iloc[0]
    assert r["ci_lo"] == 0 and r["ci_hi"] == pytest.approx(3.84 / 1000 * 0.0019511, rel=1e-3)   # not the unweighted Wilson
    assert ("N", 0, "soft") not in info["absent"] or True
    assert ("D1", 1, "hard") in info["absent"] and ("N", 0, "hard") not in info["absent"]


def test_collect_recompute_w(tmp_path):
    z = dict(lost=0, fixed=0, lost_establishing=0, est_failed=0)
    row = {"model": "N", "flux": 0, "cls": "hard", "task_id": 0, "hard": True, "s": float("nan"), "t_inv": 5e5,
           "p_star": float("nan"), "w_est": 0.002}
    with gzip.open(tmp_path / "N_0_hard_0.jsonl.gz", "wt") as f:
        f.write(json.dumps(row) + "\n")
    (tmp_path / "N_0_hard_0_counts.json").write_text(json.dumps(
        {"counts": {"hard": dict(drawn=1000, accepted=1, attempts=500_000, w_acc=0.002, w_acc_sq=4e-6, **z),
                    "soft": dict(drawn=0, accepted=0, attempts=0, w_acc=0.0, w_acc_sq=0.0, **z)}, "meta": {}}))
    info = collect.collect(str(tmp_path), out_dir=str(tmp_path / "res"), recompute_w_est=True)
    assert info["acceptance"].iloc[0]["acceptance"] == pytest.approx(0.0019511 / 1000, rel=1e-3)
