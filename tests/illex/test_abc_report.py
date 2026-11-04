"""illex.abc.abc_report: spec dump, prior-edge rule, staleness check and report formatting."""
import json

import pandas as pd

from illex.abc import abc_report as R
from illex.abc.constants import GAMMA_MAX, GAMMA_MIN, PRIOR


def _row(param, median, q025, q975, lo, hi):
    return dict(parameter=param, median=median, q025=q025, q975=q975, prior_lo=lo, prior_hi=hi)


def test_spec_bounds_come_from_constants():
    s = R.spec()
    assert (s["params"]["t_inv"]["lo"], s["params"]["t_inv"]["hi"]) == PRIOR["t_inv"]
    assert (s["params"]["p0"]["lo"], s["params"]["p0"]["hi"]) == PRIOR["p0"]
    assert (s["params"]["gamma_eff"]["lo"], s["params"]["gamma_eff"]["hi"]) == (GAMMA_MIN, GAMMA_MAX)
    assert s["p_hard"] == PRIOR["p_hard"]
    assert s["fit_keys"][-1] == "p_obs" and "tajd_I" not in s["fit_keys"]
    assert s["params"]["p0"]["log"] and not s["params"]["t_inv"]["log"]


def test_edge_flag_linear_log_and_natural_bound():
    assert R.edge_flag(_row("t_inv", 8e5, 4e5, 1.2e6, 1e5, 1.5e6)) == ""
    assert R.edge_flag(_row("t_inv", 8e5, 1.1e5, 1.2e6, 1e5, 1.5e6)) == "lower"       # q025 in the bottom 2%
    assert R.edge_flag(_row("t_inv", 1.45e6, 1e6, 1.49e6, 1e5, 1.5e6)) == "upper"
    # p0 on a log scale: q975 0.099 is at 0.998 of log10 [1e-3, 0.1]; on a linear scale 0.05 would not be lower
    assert R.edge_flag(_row("p0", 0.03, 0.005, 0.099, 1e-3, 0.1)) == "upper"
    assert R.edge_flag(_row("p0", 0.0012, 0.00105, 0.004, 1e-3, 0.1)) == "lower"
    # t_sel / t_stop: lower bound 0 is natural (edge_lo False), so no lower flag
    assert R.edge_flag(_row("t_sel", 1e4, 100.0, 5e5, 0.0, 1.5e6), params=R.PARAMS) == ""
    assert R.edge_flag(_row("t_sel", 1.45e6, 1e6, 1.49e6, 0.0, 1.5e6), params=R.PARAMS) == "upper"
    assert R.edge_flag(dict(parameter="hard_share_acc", median=0.1, q025=float("nan"), q975=float("nan"),
                            prior_lo=float("nan"), prior_hi=float("nan"))) == ""


def _raw_dir(tmp_path, spec):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "spec_used.json").write_text(json.dumps(spec))
    (d / "run_meta.json").write_text(json.dumps(dict(accuracy_grouped_cv_weighted=0.5,
                                                     accuracy_oob_unweighted_abcrf=0.6)))
    pd.DataFrame(dict(model=["N", "B"], flux=[0, 0], acc=[1e-3, 5e-3], prob_stats=[0.4, 0.6],
                      prob_total=[0.12, 0.88])).to_csv(d / "modelchoice_raw.csv", index=False)
    pd.DataFrame([dict(model="N", flux=0, parameter="p0", scale="log10", n_train=250, median=0.06, q025=0.003,
                       q25=0.04, q75=0.08, q975=0.099, prior_lo=1e-3, prior_hi=0.1),
                  dict(model="N", flux=0, parameter="t_inv", scale="linear", n_train=280, median=8e5, q025=5e5,
                       q25=7e5, q75=9e5, q975=1.2e6, prior_lo=1e5, prior_hi=1.5e6)]).to_csv(
        d / "posteriors_raw.csv", index=False)
    pd.DataFrame(dict(model=["N"], flux=[0], hard_share_stats=[0.2], hard_share_total=[1e-5])).to_csv(d / "hardshare_raw.csv", index=False)
    pd.DataFrame(dict(true=["N", "N", "B", "B"], predicted=["N", "B", "N", "B"], n=[9, 1, 2, 8],
                      w=[0.9, 0.1, 0.2, 0.8])).to_csv(d / "confusion_cv.csv", index=False)
    pd.DataFrame(dict(model=["N"], flux=[0], parameter=["p0"], n_pods=[200], cover50=[0.5],
                      cover95=[0.95])).to_csv(d / "coverage_raw.csv", index=False)
    pd.DataFrame(dict(model=["N"], flux=[0], parameter=["p0"], tol=[0.4], n_accept=[100], rf_median=[0.06],
                      rejection_median=[0.05], loclinear_median=[None], note=["outside"])).to_csv(
        d / "crosscheck_raw.csv", index=False)
    pd.DataFrame(dict(model=["N"], flux=[0], n_draws=[1000], expected_hard=[0.1], drawn_hard=[0],
                      unique_rows=[120])).to_csv(d / "draws_raw.csv", index=False)
    (d / "abc_post_N_0.csv").write_text("# NOTE: x\nmodel,cls\nN,soft\n")
    return d


def test_report_writes_tables_and_flags(tmp_path):
    d = _raw_dir(tmp_path, json.loads(json.dumps(R.spec())))
    out = tmp_path / "out"
    R.report(str(d), str(out))
    mc = pd.read_csv(out / "abc_modelchoice.csv")
    assert list(mc.p_now_term) == list(mc.acc)
    post = pd.read_csv(out / "abc_posteriors.csv")
    assert post.loc[post.parameter == "p0", "edge"].iloc[0] == "upper"
    assert pd.isna(post.loc[post.parameter == "t_inv", "edge"].iloc[0])
    assert set(post.parameter) == {"p0", "t_inv", "hard_share_noacc", "hard_share_acc"}
    conf = pd.read_csv(out / "abc_confusion.csv")
    assert set(conf.method) == {"grouped_cv"}
    txt = (out / "abc_modelchoice.txt").read_text()
    assert "HEADLINE" in txt and "Prior-edge flags: N/p0: upper" in txt and "GLOBAL" in txt
    assert "WARNING" not in txt
    assert (out / "abc_post_N_0.csv").read_text().startswith("# NOTE")
    assert (out / "abc_draws_summary.csv").exists()


def test_report_warns_on_stale_spec(tmp_path, capsys):
    stale = json.loads(json.dumps(R.spec()))
    stale["params"]["p0"]["hi"] = 0.5
    d = _raw_dir(tmp_path, stale)
    R.report(str(d), str(tmp_path / "out"))
    assert "differs from illex/abc/constants.py" in capsys.readouterr().out
    assert "WARNING" in (tmp_path / "out" / "abc_modelchoice.txt").read_text()
    # the edge check uses the prior_lo / prior_hi R wrote on each posterior row (the run's own bounds)
    post = pd.read_csv(tmp_path / "out" / "abc_posteriors.csv")
    assert post.loc[post.parameter == "p0", "edge"].iloc[0] == "upper"
