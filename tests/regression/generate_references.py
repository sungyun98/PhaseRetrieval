"""Generate regression reference files.

Example (references from the original code, in the legacy environment):

    conda run -n pr-legacy python tests/regression/generate_references.py \
        --impl pr_legacy --code-root ../_legacy/PhaseRetrieval --out tests/regression/references

``--impl`` selects ``adapters/<impl>.py``; ``--code-root`` is the checkout whose code is
imported (for references: a worktree at tag v1.0-legacy).
"""

import argparse
import datetime
import importlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402


def git_describe(path):
    try:
        out = subprocess.run(
            ["git", "-C", path, "describe", "--tags", "--always", "--dirty"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        sha = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        return "{} ({})".format(out.stdout.strip(), sha.stdout.strip())
    except OSError:
        return "unknown"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--impl", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases-module", default="cases_pr")
    p.add_argument("--only", nargs="*", help="run only these case names")
    args = p.parse_args()

    adapter_mod = importlib.import_module("adapters." + args.impl)
    api = adapter_mod.Adapter(args.code_root)
    cases = importlib.import_module(args.cases_module)
    os.makedirs(args.out, exist_ok=True)
    base_meta = {
        "adapter": api.name,
        "adapter_notes": list(getattr(api, "notes", [])),
        "code": git_describe(args.code_root),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "environment": harness.environment(),
    }
    for name, case in cases.CASES.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        results = case(api)
        meta = dict(base_meta, case=name, seconds=round(time.time() - t0, 2))
        harness.save(os.path.join(args.out, name), results, meta)
        errs = [k for k, v in results.items() if isinstance(v, str)]
        print(
            "{:28s} {:6.1f} s  {}".format(
                name,
                meta["seconds"],
                ", ".join("{}={}".format(k, results[k]) for k in errs) or "ok",
            )
        )


if __name__ == "__main__":
    main()
