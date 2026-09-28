# Numerical regression suite

Checks that changes to the code do not alter numerical results beyond floating-point
tolerance. The reference outputs in `references/` were generated from the original code
(tag `v1.0-legacy`) with PyTorch 1.6 on CPU; see `environment-legacy.yml`.

## Layout

| File | Purpose |
| --- | --- |
| `cases_pr.py` | Test cases: every algorithm (HIO, RAAR, gRAAR, dRAAR, GPS-R/F, dpGPS-R/F, shrinkwrap, k-space output) and utility (support generation, shifts, amplitude/phase, masks, Gaussian smoothing, preconditioner, subpixel alignment, pairwise distance, PRTF, PSD, SVD eigenmodes), with tolerances |
| `adapters/` | One adapter per implementation, translating between NumPy arrays and that implementation's API |
| `harness.py` | Saving, loading and comparing results |
| `generate_references.py` | Writes `references/<case>.npz` and `references/<case>.json` |
| `test_regression.py` | pytest entry point comparing an implementation with the references |

Inputs come from `sample_lena.mat` and from `numpy.random.RandomState(seed)`, whose streams
are stable across NumPy versions, so every implementation is tested on identical data.
Iterative algorithms run 2 random initial phases for 40 iterations; real-space outputs are
stored as the 64 x 64 region containing the support (everything outside it must be zero,
which is checked through `u_outside_abs_sum`).

## Running

```bash
# current code (adapters/pr_modern.py)
REG_IMPL=pr_modern REG_CODE_ROOT=. conda run -n <env> python -m pytest tests/regression -q
# float64 equivalence check against references_f64/
REG_FLOAT64=1 REG_IMPL=pr_modern REG_CODE_ROOT=. conda run -n <env> python -m pytest tests/regression -q

# the original code (reproduces the references bit for bit)
git worktree add --detach ../_legacy/PhaseRetrieval v1.0-legacy
REG_IMPL=pr_legacy REG_CODE_ROOT=../_legacy/PhaseRetrieval \
    conda run -n pr-legacy python -m pytest tests/regression -q
```

A result passes when its relative L2 difference `||new - ref|| / ||ref||` is within the
case tolerance in `cases_pr.py` (1e-6 for direct operations, 1e-5 for the preconditioner
network, 1e-4 for iterative algorithms). NaN positions and error types must match exactly.

## Float32 noise and the float64 check

Iterative phase retrieval amplifies float32 rounding: the legacy code's own float32 results
differ from its float64 results by up to 4e-2 (gRAAR, 40 iterations), so two correct
implementations can differ by that much in float32. `generate_f64_references.py` runs the
legacy code in float64 (`references_f64/`, iterative cases and the preconditioner) and stores
this noise per result; the float32 tolerance of a result is raised to three times it. The
float64 run (`REG_FLOAT64=1`) must match `references_f64/` to 1e-9, which checks that the
algorithms are the same independently of float32 rounding.

## Known behaviour of the original code recorded in the references

- `pr_HIO_shrinkwrap`, `pr_GPS-R_shrinkwrap`: the original `ShrinkWrap.forward` raises
  `TypeError: conv2d() got an unexpected keyword argument 'padding_mode'`.
- The original `Preconditioner` loads weights saved on `cuda:0` without `map_location`; the
  legacy adapter makes `torch.load` default to `map_location='cpu'` so it runs on CPU.

Intended changes of results are listed with their reason in `EXPECTED_CHANGES`
(`cases_pr.py`); for implementations other than the one that produced the references, such
cases are reported as expected failures.
