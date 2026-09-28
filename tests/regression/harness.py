"""Shared machinery for the numerical regression suite.

A *case* is a function ``case(api, rng) -> dict`` that runs one algorithm or utility through
an implementation adapter (``api``) and returns named results. Results are NumPy arrays,
Python scalars, or the string produced by :func:`capture_error` when the implementation
raises. References are generated once from the original code (tag ``v1.0-legacy``) and
later implementations are compared against them with :func:`compare`.

Kept compatible with Python 3.7 so it runs in the legacy environment.
"""
import json
import platform
import sys
import traceback

import numpy as np

ERROR_PREFIX = "ERROR: "


def capture_error(fn, *args, **kwargs):
    """Run ``fn`` and return its result, or ``'ERROR: <type>: <message>'`` if it raises."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # the legacy code is expected to fail in documented cases
        last = traceback.extract_tb(exc.__traceback__)[-1]
        return "{}{}: {} [{}:{}]".format(ERROR_PREFIX, type(exc).__name__, exc,
                                         last.filename.split("/")[-1], last.lineno)


def save(path, results, meta):
    """Save one case: arrays go to ``<path>.npz``, everything else to ``<path>.json``."""
    arrays = {k: np.asarray(v) for k, v in results.items() if isinstance(v, np.ndarray)}
    other = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(str(path) + ".npz", **arrays)
    with open(str(path) + ".json", "w") as f:
        json.dump({"meta": meta, "values": other, "arrays": sorted(arrays)}, f, indent=1,
                  default=_jsonable)


def load(path):
    with open(str(path) + ".json") as f:
        doc = json.load(f)
    out = dict(doc["values"])
    with np.load(str(path) + ".npz") as z:
        for k in z.files:
            out[k] = z[k]
    return out, doc["meta"]


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(type(x))


def rel_l2(a, b):
    """Relative L2 difference ||a - b|| / max(||b||, tiny)."""
    a = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else np.float64)
    b = np.asarray(b, dtype=a.dtype)
    den = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (den if den > 0 else 1e-300))


def compare(ref, new, tol):
    """Compare two result dicts. Returns a list of human-readable failure strings.

    ``tol`` maps result names (or ``'*'``) to the maximum allowed relative L2 difference.
    Error strings and non-array values must match exactly (error text is compared by type).
    """
    failures = []
    for key in sorted(set(ref) | set(new)):
        if key not in new:
            failures.append("{}: missing in new results".format(key))
            continue
        if key not in ref:
            failures.append("{}: not in reference".format(key))
            continue
        r, n = ref[key], new[key]
        r_err = isinstance(r, str) and r.startswith(ERROR_PREFIX)
        n_err = isinstance(n, str) and n.startswith(ERROR_PREFIX)
        if r_err or n_err:
            if not (r_err and n_err and r.split(":")[1] == n.split(":")[1]):
                failures.append("{}: reference {!r} vs new {!r}".format(key, r, n))
            continue
        if isinstance(r, np.ndarray) or isinstance(n, np.ndarray):
            r, n = np.asarray(r), np.asarray(n)
            if r.shape != n.shape:
                failures.append("{}: shape {} vs {}".format(key, r.shape, n.shape))
                continue
            if r.dtype == bool or n.dtype == bool:
                bad = int(np.sum(r != n))
                if bad:
                    failures.append("{}: {} boolean elements differ".format(key, bad))
                continue
            limit = tol.get(key, tol.get("*", 0.0))
            if r.dtype.kind in "fc" or n.dtype.kind in "fc":
                # NaN is a legitimate output (e.g. PSD radii without valid pixels): positions must match
                r_nan, n_nan = np.isnan(r), np.isnan(n)
                if np.any(r_nan != n_nan):
                    failures.append("{}: NaN positions differ ({} vs {})".format(key, int(r_nan.sum()), int(n_nan.sum())))
                    continue
                r, n = r[~r_nan], n[~n_nan]
            err = rel_l2(n, r)
            if not err <= limit:
                failures.append("{}: rel L2 diff {:.3e} > {:.1e}".format(key, err, limit))
        else:
            if isinstance(r, float) or isinstance(n, float):
                limit = tol.get(key, tol.get("*", 0.0))
                if not abs(float(n) - float(r)) <= limit * max(abs(float(r)), 1e-300):
                    failures.append("{}: {!r} vs {!r}".format(key, r, n))
            elif _norm(r) != _norm(n):
                failures.append("{}: {!r} vs {!r}".format(key, r, n))
    return failures


def _norm(x):
    return json.loads(json.dumps(x, default=_jsonable))


def environment():
    """Versions recorded with each reference."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "scipy", "skimage", "torch", "torchvision", "PIL", "h5py"]:
        try:
            info[name] = __import__(name).__version__
        except Exception:
            pass
    return info
