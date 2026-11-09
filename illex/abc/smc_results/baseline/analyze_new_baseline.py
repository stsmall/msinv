"""Summarise ppc_new_baseline.py replicates with ppc/ppc_analyze.py, scored against abc_observed_ctl2.json (the
observed vector divided by the representative control) instead of abc_observed.json.

    PYTHONPATH=<checkout> python analyze_new_baseline.py ppc_newbase_B_f0.csv ppc_newbase_B_f1.csv
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ppc_analyze", HERE.parent / "ppc" / "ppc_analyze.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)
pa.OBS = HERE.parents[3] / "results" / "illex" / "abc_observed_ctl2.json"
pa.main(sys.argv[1:])
