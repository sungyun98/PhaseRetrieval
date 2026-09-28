"""Regression cases for PhaseRetrieval (canonical NumPy interface, see adapters/).

Each case returns a dict of results. Inputs are derived from ``sample_lena.mat`` or from
``numpy.random.RandomState(seed)``, whose streams are stable across NumPy versions.
``TOLERANCE`` gives the maximum relative L2 difference accepted against the references;
``EXPECTED_CHANGES`` documents cases whose results are allowed (and expected) to differ
from the legacy references, with the reason.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import fourier_shift

from harness import capture_error

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
N_SEED = 2
N_ITER = 40
CROP = (slice(224, 288), slice(224, 288))  # contains the lena support (rows 239-272, cols 242-268)
LOWFREQ = (slice(192, 320), slice(192, 320))  # central k-space block of a 512 x 512 frame


def load_lena():
    f = loadmat(os.path.join(REPO, "sample_lena.mat"))
    intensity = f["intensity"].astype(np.float32)
    missing = np.isnan(intensity)
    intensity[missing] = 0
    support = f["support"] > 0
    obj = f["sample"] * support
    obj[obj < 0] = 0
    # same preparation as main.ipynb: ifftshifted amplitude and missing mask
    amplitude = np.sqrt(np.fft.ifftshift(intensity)).astype(np.float32)
    unknown = np.fft.ifftshift(missing).astype(np.float32)
    return dict(intensity=intensity, missing=missing, support=support.astype(np.float32),
                obj=obj.astype(np.float32), amplitude=amplitude, unknown=unknown)


def random_phase(seed, shape):
    theta = np.random.RandomState(seed).rand(*shape) * 2 * np.pi
    return np.exp(1j * theta).astype(np.complex64)


# ---- func.py ---------------------------------------------------------------------------
def case_func_make_support(api):
    d = load_lena()
    rect = api.make_support(d["intensity"], type="rect", radius=(8, 4))
    auto = api.make_support(d["intensity"], type="auto", threshold=0.2)
    return {"rect": np.asarray(rect, dtype=np.float32), "auto": np.asarray(auto, dtype=np.float32)}


def case_func_shifts(api):
    rs = np.random.RandomState(1)
    out = {}
    for tag, shape in [("even", (2, 6, 8)), ("odd", (2, 5, 7))]:
        x = rs.randn(*shape).astype(np.float32)
        z = (rs.randn(*shape) + 1j * rs.randn(*shape)).astype(np.complex64)
        out["fftshift_real_" + tag] = api.fftshift(x)
        out["ifftshift_real_" + tag] = api.ifftshift(x)
        out["fftshift_complex_" + tag] = api.fftshift(z)
        out["ifftshift_complex_" + tag] = api.ifftshift(z)
    return out


def case_func_amplitude_phase(api):
    rs = np.random.RandomState(2)
    z = (rs.randn(2, 6, 8) + 1j * rs.randn(2, 6, 8)).astype(np.complex64)
    z[0, 0, :3] = 0  # exercise the zero-amplitude branch of phase()
    return {"amplitude": api.amplitude(z), "phase": api.phase(z)}


def case_func_sqmesh_freqfilter(api):
    return {"sqmesh_5x6": api.sqmesh(5, 6), "sqmesh_8x8": api.sqmesh(8, 8),
            "freqfilter_512_10": np.asarray(api.freqfilter(512, 10), dtype=np.float64),
            "freqfilter_64_3": np.asarray(api.freqfilter(64, 3), dtype=np.float64)}


def case_func_gaussian_smoothing(api):
    d = load_lena()
    masked = api.gaussian_smoothing(d["intensity"][None], 1.5, mask=(1 - d["missing"][None]).astype(np.float32))
    x = np.random.RandomState(3).rand(1, 40, 36).astype(np.float32)
    plain = api.gaussian_smoothing(x, 2.3)
    return {"masked_sigma1.5": masked, "plain_sigma2.3": plain}


# ---- preconditioner.py -------------------------------------------------------------------
def case_preconditioner(api):
    d = load_lena()
    a, u = d["amplitude"], d["unknown"]
    return {"kernel_deep_limit0.28": api.preconditioner(a, u, 0.28, deep=True),
            "kernel_nondeep_limit0.28": api.preconditioner(a, u, 0.28, deep=False),
            "kernel_deep_nolimit": api.preconditioner(a, u, 0.0, deep=True),
            "denoised_limit0.28": api.preconditioner(a, u, 0.28, deep=True, toggle=True)}


# ---- phaseretrieval.py -------------------------------------------------------------------
GPS_COMMON = dict(sigma=(0, 0.01, 0.4, 0.1, 0.7, 1), alpha_count=10, t=1, s=0.8)
SHRINKWRAP = dict(shrinkwrap=True, sigma_initial=3, sigma_limit=1.5, ratio_update=0.01,
                  threshold=0.1, interval=10)
PR_CONFIGS = {
    "HIO": dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0.2),
    "RAAR_step": dict(algorithm="RAAR", error="R", beta=0.75, beta_type="step", beta_lim=1, boundary_push=0),
    "RAAR_linear_NLL": dict(algorithm="RAAR", error="NLL", beta=0.5, beta_type="linear", beta_lim=1,
                            boundary_push=0.1),
    "RAAR_schedule": dict(algorithm="RAAR", error="R", beta=(0, 0.5, 0.5, 0.9), beta_type="const",
                          boundary_push=0),
    "gRAAR": dict(algorithm="gRAAR", error="R", beta=0.5, beta_type="linear", beta_lim=1, boundary_push=0),
    "dRAAR": dict(algorithm="dRAAR", error="R", beta=0.5, beta_type="linear", beta_lim=1, boundary_push=0,
                  limit=0.28, deep=True),
    "GPS-R": dict(algorithm="GPS-R", error="R", **GPS_COMMON),
    "GPS-F": dict(algorithm="GPS-F", error="R", **GPS_COMMON),
    "dpGPS-R": dict(algorithm="dpGPS-R", error="R", limit=0.28, deep=True, **GPS_COMMON),
    "dpGPS-F": dict(algorithm="dpGPS-F", error="R", limit=0.28, deep=True, **GPS_COMMON),
    "HIO_shrinkwrap": dict(algorithm="HIO", error="R", beta=0.9, beta_type="const", boundary_push=0.2,
                           **SHRINKWRAP),
    "GPS-R_shrinkwrap": dict(algorithm="GPS-R", error="R", **dict(GPS_COMMON, **SHRINKWRAP)),
}
TOGGLE_CASES = ["HIO", "GPS-F"]  # also run with toggle=True (k-space output)


def run_pr(api, config_name, toggle=False, seed=10):
    d = load_lena()
    info = dict(error="R", shrinkwrap=False)
    info.update(PR_CONFIGS[config_name])
    h, w = d["amplitude"].shape
    phase = random_phase(seed, (N_SEED, h, w))

    def run():
        return api.phase_retrieval(d["amplitude"], d["support"], d["unknown"], info, N_ITER, phase, toggle=toggle)

    res = capture_error(run)
    if isinstance(res, str):
        return {"result": res}
    out, path = res
    if toggle:
        low = np.fft.fftshift(out[0])[LOWFREQ]  # central 128 x 128 block (used by the float64 check)
        return {"z_seed0": out[0], "z_lowfreq_seed0": low, "path": path}
    outside = np.array(out)
    outside[(slice(None),) + CROP] = 0
    return {"u_crop": out[(slice(None),) + CROP], "u_outside_abs_sum": float(np.abs(outside).sum()), "path": path}


def make_pr_case(name, toggle):
    def case(api):
        if toggle and not getattr(api, "supports_toggle", True):
            return {"result": "SKIPPED: toggle not supported"}
        return run_pr(api, name, toggle=toggle)
    case.__name__ = "case_pr_{}{}".format(name, "_z" if toggle else "")
    return case


# ---- eval.py ---------------------------------------------------------------------------
def aligned_stack():
    """Five shifted/flipped/noisy copies of the lena object (64 x 64 crop)."""
    obj = load_lena()["obj"][CROP].astype(np.float64)
    rs = np.random.RandomState(4)

    def shift(x, s):
        return np.fft.ifft2(fourier_shift(np.fft.fft2(x), s)).real

    stack = [obj, shift(obj, (1.3, -2.1)), np.flip(shift(obj, (0.5, 0.7))), shift(obj, (-3, 2)), np.flip(obj)]
    stack = np.stack(stack) + 0.002 * rs.randn(5, *obj.shape)
    return stack.astype(np.float64), obj


def case_eval_subpixel_alignment(api):
    stack, obj = aligned_stack()
    error = np.array([0.3, 0.1, 0.5, 0.2, 0.4])
    first, err_sorted = api.subpixel_alignment(stack, error=error, subpixel=10)
    ref1 = api.subpixel_alignment(stack, ref=obj, subpixel=1)
    ref10 = api.subpixel_alignment(stack, ref=obj, subpixel=10)
    return {"to_first_sub10": first, "error_sorted": err_sorted, "to_ref_sub1": ref1, "to_ref_sub10": ref10}


def case_eval_metrics(api):
    stack, obj = aligned_stack()
    aligned = api.subpixel_alignment(stack, ref=obj, subpixel=10)
    ref = np.abs(np.fft.fftshift(np.fft.fft2(obj)))
    mask = np.random.RandomState(5).rand(*obj.shape) < 0.05
    d = load_lena()
    return {"pairwise_distance": api.pairwise_distance(aligned),
            "prtf": api.prtf(aligned, ref, mask=mask),
            "prtf_nomask": api.prtf(aligned, ref),
            "psd_intensity_masked": api.psd(d["intensity"], mask=d["missing"]),
            "psd_random": api.psd(np.random.RandomState(6).rand(40, 36))}


def case_eval_eigenmode(api):
    stack, _ = aligned_stack()
    modes, s = api.eigenmode(stack)
    modes3, s3 = api.eigenmode(stack, k=3, lowrank=False)
    modes3b, s3b, approx = api.eigenmode(stack, k=3, lowrank=True)
    # singular vectors are defined up to sign; normalize so the largest-|.| element is positive
    def fix(m):
        m = np.array(m)
        for i in range(m.shape[0]):
            flat = m[i].ravel()
            if flat[np.argmax(np.abs(flat))] < 0:
                m[i] = -m[i]
        return m
    return {"modes": fix(modes), "singular": s, "modes_k3": fix(modes3), "singular_k3": s3,
            "modes_k3_lowrank": fix(modes3b), "singular_k3_lowrank": s3b, "lowrank_approx": approx}


# ---- registry ----------------------------------------------------------------------------
CASES = {}
for _fn in [case_func_make_support, case_func_shifts, case_func_amplitude_phase, case_func_sqmesh_freqfilter,
            case_func_gaussian_smoothing, case_preconditioner]:
    CASES[_fn.__name__[5:]] = _fn
for _name in PR_CONFIGS:
    CASES["pr_" + _name] = make_pr_case(_name, False)
for _name in TOGGLE_CASES:
    CASES["pr_{}_z".format(_name)] = make_pr_case(_name, True)
for _fn in [case_eval_subpixel_alignment, case_eval_metrics, case_eval_eigenmode]:
    CASES[_fn.__name__[5:]] = _fn

TOLERANCE = {name: {"*": 1e-6} for name in CASES}
TOLERANCE["preconditioner"] = {"*": 1e-5}
for _name in CASES:
    if _name.startswith("pr_"):
        TOLERANCE[_name] = {"*": 1e-4}

# Cases with float64 references (references_f64/): iterative algorithms and the preconditioner.
F64_CASES = [n for n in CASES if n.startswith("pr_") and "shrinkwrap" not in n] + ["preconditioner"]
F64_DROP = ["z_seed0"]  # full k-space arrays are not stored in float64 (z_lowfreq_seed0 is)
F64_TOLERANCE = 1e-9
F32_NOISE_FACTOR = 3  # float32 tolerance >= this factor x the legacy code's own float32 error

_SW_FIX = ("the original ShrinkWrap.forward raises TypeError (padding_mode is not an F.conv2d argument); "
           "the native-complex port pads with F.pad(..., mode='reflect') as in DPR, and the ShrinkWrap "
           "Gaussian kernel is centred instead of ifftshifted (the ifftshifted kernel split the Gaussian "
           "into lobes about +-ceil(2 * sigma_initial) px apart and enlarged the support)")
EXPECTED_CHANGES = {
    "pr_HIO_shrinkwrap": _SW_FIX,
    "pr_GPS-R_shrinkwrap": _SW_FIX,
}
