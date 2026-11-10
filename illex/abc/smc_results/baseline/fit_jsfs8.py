"""baseline_refit.py fit with the 8 informative control JSFS cells. The 4 cells for a variant near-absent in one
group and near-fixed in the other (jsfs_03, 04, 13, 14) are ~1e-9 to 1e-14 in a collinear region (numerical mass
from the projection), so their MAD-scaled distance swamps every other statistic.

    PYTHONNOUSERSITE=1 PYTHONPATH=<checkout> python fit_jsfs8.py --draws draws.csv --observed <json> --out fit_jsfs8.json --with-jsfs
"""
import sys
from pathlib import Path

import illex

sys.path.insert(0, str(Path(illex.__file__).resolve().parent / "abc" / "smc_results" / "baseline"))
import baseline_refit as B  # noqa: E402

B.JSFS = [c for c in B.JSFS if c not in ("jsfs_03", "jsfs_04", "jsfs_13", "jsfs_14")]
sys.argv = [sys.argv[0], "fit"] + sys.argv[1:]
B.main()
