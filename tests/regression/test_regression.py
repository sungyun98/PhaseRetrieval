"""Compare an implementation against the stored references.

    # float32 (default): compare with references/
    REG_IMPL=pr_modern REG_CODE_ROOT=. conda run -n <env> python -m pytest tests/regression -q
    # float64: compare with references_f64/ (strict equivalence check of the algorithms)
    REG_FLOAT64=1 REG_IMPL=pr_modern REG_CODE_ROOT=. conda run -n <env> python -m pytest tests/regression -q

``REG_IMPL`` selects ``adapters/<impl>.py`` and ``REG_CODE_ROOT`` the checkout to import.

Float32 tolerance for an array is the case tolerance, raised to ``F32_NOISE_FACTOR`` times the
float32 rounding noise of the reference code itself when that noise was measured (stored in
``references_f64/<case>.json``). Iterative phase retrieval amplifies rounding differences, so
float32 results of two correct implementations differ by about that much.
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
REFS_F64 = os.environ.get("REG_REFERENCES_F64", os.path.join(HERE, "references_f64"))
F64 = os.environ.get("REG_FLOAT64") == "1"


@pytest.fixture(scope="module")
def api():
    impl = os.environ.get("REG_IMPL")
    root = os.environ.get("REG_CODE_ROOT")
    if not impl or not root:
        pytest.skip("set REG_IMPL and REG_CODE_ROOT to run the regression suite")
    return importlib.import_module("adapters." + impl).Adapter(root)


def float32_tolerance(name):
    tol = dict(cases.TOLERANCE.get(name, {"*": 0.0}))
    path = os.path.join(REFS_F64, name)
    if os.path.exists(path + ".json"):
        _, meta = harness.load(path)
        factor = getattr(cases, "F32_NOISE_FACTOR", 3)
        for key, noise in meta.get("f32_reference_rel_error", {}).items():
            tol[key] = max(tol.get(key, tol.get("*", 0.0)), factor * noise)
    return tol


def check_expected(name, failures, api, ref_meta):
    # EXPECTED_CHANGES apply to later implementations; the reference implementation must reproduce itself
    if name in cases.EXPECTED_CHANGES and ref_meta.get("adapter") != api.name:
        reason = cases.EXPECTED_CHANGES[name]
        if failures:
            pytest.xfail("expected change ({}): {}".format(reason, "; ".join(failures)))
        pytest.fail("marked as expected change ({}) but matches the reference".format(reason))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("name", list(cases.CASES))
def test_case(api, name):
    if F64:
        pytest.skip("float32 comparison is not run in float64 mode")
    ref, meta = harness.load(os.path.join(REFS, name))
    new = cases.CASES[name](api)
    check_expected(name, harness.compare(ref, new, float32_tolerance(name)), api, meta)


@pytest.mark.parametrize("name", list(getattr(cases, "F64_CASES", [])))
def test_case_float64(api, name):
    if not F64:
        pytest.skip("set REG_FLOAT64=1 to run the float64 equivalence check")
    ref, meta = harness.load(os.path.join(REFS_F64, name))
    new = cases.CASES[name](api)
    check_expected(name, harness.compare(ref, new, {"*": cases.F64_TOLERANCE}, subset=True), api, meta)
