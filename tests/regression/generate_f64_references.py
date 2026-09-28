"""Generate float64 references and measure the float32 noise of the reference code.

Runs the cases listed in ``F64_CASES`` in double precision (``REG_FLOAT64=1``) and stores
them in ``references_f64/``. For every array it also records, in the metadata, the relative
L2 difference between the float32 reference and this float64 result: the rounding noise
of the reference code itself. ``test_regression.py`` uses it to set float32 tolerances.

    conda run -n pr-legacy python tests/regression/generate_f64_references.py \
        --impl pr_legacy --code-root ../_legacy/PhaseRetrieval
"""

import argparse
import importlib
import os
import sys

os.environ["REG_FLOAT64"] = "1"  # must be set before an adapter is imported
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402
import numpy as np  # noqa: E402
from generate_references import git_describe  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--impl", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--cases-module", default="cases_pr")
    p.add_argument("--references", default=os.path.join(HERE, "references"))
    p.add_argument("--out", default=os.path.join(HERE, "references_f64"))
    args = p.parse_args()

    api = importlib.import_module("adapters." + args.impl).Adapter(args.code_root)
    cases = importlib.import_module(args.cases_module)
    os.makedirs(args.out, exist_ok=True)
    for name in cases.F64_CASES:
        results = cases.CASES[name](api)
        ref32, _ = harness.load(os.path.join(args.references, name))
        noise = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in "fc" and k in ref32:
                finite = ~np.isnan(v)
                noise[k] = harness.rel_l2(np.asarray(ref32[k])[finite], v[finite])
        for k in cases.F64_DROP:
            results.pop(k, None)
        meta = {
            "case": name,
            "adapter": api.name,
            "code": git_describe(args.code_root),
            "precision": "float64",
            "environment": harness.environment(),
            "f32_reference_rel_error": noise,
        }
        harness.save(os.path.join(args.out, name), results, meta)
        print(
            "{:24s} float32 noise: {}".format(
                name, ", ".join(f"{k}={v:.2e}" for k, v in noise.items())
            )
        )


if __name__ == "__main__":
    main()
