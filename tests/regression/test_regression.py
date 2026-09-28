"""Compare an implementation against the stored references.

    REG_IMPL=pr_legacy REG_CODE_ROOT=../_legacy/PhaseRetrieval \
        conda run -n pr-legacy python -m pytest tests/regression -q

``REG_IMPL`` selects ``adapters/<impl>.py`` and ``REG_CODE_ROOT`` the checkout to import.
``REG_REFERENCES`` overrides the reference directory (default: ``references/`` here).
"""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import harness  # noqa: E402

CASES_MODULE = os.environ.get("REG_CASES", "cases_pr")
cases = importlib.import_module(CASES_MODULE)
REFS = os.environ.get("REG_REFERENCES", os.path.join(HERE, "references"))


@pytest.fixture(scope="module")
def api():
    impl = os.environ.get("REG_IMPL")
    root = os.environ.get("REG_CODE_ROOT")
    if not impl or not root:
        pytest.skip("set REG_IMPL and REG_CODE_ROOT to run the regression suite")
    return importlib.import_module("adapters." + impl).Adapter(root)


@pytest.mark.parametrize("name", list(cases.CASES))
def test_case(api, name):
    ref, _ = harness.load(os.path.join(REFS, name))
    new = cases.CASES[name](api)
    failures = harness.compare(ref, new, cases.TOLERANCE.get(name, {"*": 0.0}))
    if name in cases.EXPECTED_CHANGES:
        reason = cases.EXPECTED_CHANGES[name]
        if failures:
            pytest.xfail("expected change ({}): {}".format(reason, "; ".join(failures)))
        pytest.fail("marked as expected change ({}) but matches the reference".format(reason))
    assert not failures, "\n".join(failures)
