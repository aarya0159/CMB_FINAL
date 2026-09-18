"""Cosmic-variance scaling campaign (Table 1, Figure 1a).

Sweeps the number of standing waves N and measures the seed-to-seed scatter
of the map's root-mean-square anisotropy, testing the random-walk prediction

    sigma_dT / <dT>  =  (sqrt(r)/2) N^(-1/2)  =  0.81 N^(-1/2)

against the simulator's own field-generation engine (cmb_simulator.py,
unmodified). Only the pygame graphics import is stubbed.

Writes variance_results.json, consumed by fit_two_component.py and
make_figure.py.

Run:  python variance_campaign.py
"""
import sys, types, json, math, time
import numpy as np
from scipy.integrate import quad

# ---- stub pygame (graphics only; CMBField never touches it) ---------------
pg = types.ModuleType("pygame")
class _Rect:
    def __init__(self, *a):
        if len(a) == 4: self.x, self.y, self.w, self.h = a
pg.Rect = _Rect
sys.modules["pygame"] = pg
import cmb_simulator as sim

W, H = sim.FIELD_W, sim.FIELD_H      # 384 x 240, the simulator's own grid
TFIX = 5.0                           # fixed instant in the standing-wave cycle

# (N, number of independently seeded universes)
PLAN = [(6, 400), (12, 400), (24, 400), (48, 300),
        (96, 200), (192, 120), (384, 80)]

def seed_for(k, N):
    """Deterministic seed for the kth realisation at wave count N."""
    return 20_000 + 37 * k + N

# ---------------------------------------------------------------------------
# Predicted coefficients, computed from the code's amplitude distribution
# BEFORE any fitting. Wavelengths are log-uniform on [18, 260] px; the
# amplitude is env(lambda) * u with env = exp(-[ln(lambda/70)/0.75]^2) and
# u ~ U(0.6, 1.4). The temporal factor contributes cos^2 with E=1/2, E=3/8.
#
#   q = E[a^4] / E[a^2]^2          shape of the squared-amplitude distribution
#   r = 3q/2 - 1                   relative variance of one wave's contribution
#   naive coefficient  = sqrt(r)/2     (normalisation ignored)
#   refined coefficient A = q/8        (normalisation retained)
# ---------------------------------------------------------------------------
LAM_LO, LAM_HI, PEAK, WID = 18.0, 260.0, 70.0, 0.75
U_LO, U_HI = 0.6, 1.4

def q_by_quadrature():
    lo, hi, ps = math.log(LAM_LO), math.log(LAM_HI), math.log(PEAK)
    L = hi - lo
    E = lambda p: quad(lambda y: math.exp(-p * ((y - ps) / WID) ** 2),
                       lo, hi, epsabs=1e-13)[0] / L
    e2, e4 = E(2), E(4)
    eu2 = (U_HI**3 - U_LO**3) / (3 * (U_HI - U_LO))
    eu4 = (U_HI**5 - U_LO**5) / (5 * (U_HI - U_LO))
    return (e4 / e2**2) * (eu4 / eu2**2)

def q_by_sampling(m=4_000_000, seed=7):
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.uniform(math.log(LAM_LO), math.log(LAM_HI), m))
    a2 = (np.exp(-((np.log(lam / PEAK)) / WID) ** 2) * rng.uniform(U_LO, U_HI, m)) ** 2
    return float((a2 ** 2).mean() / a2.mean() ** 2)

q_quad, q_mc = q_by_quadrature(), q_by_sampling()
r_quad = 1.5 * q_quad - 1.0
print(f"q (quadrature) = {q_quad:.5f}   q (4M samples) = {q_mc:.5f}")
print(f"r = 3q/2 - 1   = {r_quad:.5f}")
print(f"naive coefficient sqrt(r)/2 = {math.sqrt(r_quad)/2:.4f}")
print(f"refined coefficient A = q/8 = {q_quad/8:.4f}\n")

# ---------------------------------------------------------------- measurement
results = {}
for N, nseeds in PLAN:
    t0 = time.time()
    stds = np.array([float(sim.CMBField(W, H, n_waves=N, seed=seed_for(k, N))
                           .evaluate(TFIX).std()) for k in range(nseeds)])
    frac = stds.std(ddof=1) / stds.mean()
    err = frac / math.sqrt(2 * nseeds - 1)   # standard error of a sample s.d.
    results[N] = {"nseeds": nseeds, "mean_std": float(stds.mean()),
                  "frac_scatter": float(frac), "frac_err": float(err)}
    print(f"N={N:4d}  universes={nseeds:4d}  scatter={frac:.4f} +- {err:.4f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

print(f"\ntotal universes: {sum(n for _, n in PLAN)}")

# ------------------------------------------- weighted power-law fit in log space
Ns = np.array(sorted(results))
y = np.array([results[n]["frac_scatter"] for n in Ns])
ey = np.array([results[n]["frac_err"] for n in Ns])
lx, ly, w = np.log(Ns), np.log(y), (y / ey) ** 2
Wm = w.sum(); xb = (w * lx).sum() / Wm; yb = (w * ly).sum() / Wm
slope = (w * (lx - xb) * (ly - yb)).sum() / (w * (lx - xb) ** 2).sum()
slope_err = 1.0 / math.sqrt((w * (lx - xb) ** 2).sum())
print(f"\nmeasured exponent = {slope:.4f} +- {slope_err:.4f}   (prediction -0.5)")
print(f"  distance from -0.5: {abs(-0.5-slope)/slope_err:.1f} sigma")

json.dump({"q_quadrature": q_quad, "q_sampled": q_mc, "r": r_quad,
           "naive_coeff": math.sqrt(r_quad) / 2, "A_theory": q_quad / 8,
           "grid": [W, H], "t_fixed": TFIX,
           "seed_rule": "20000 + 37k + N",
           "results": {str(k): v for k, v in results.items()},
           "exponent": {"slope": slope, "slope_err": slope_err}},
          open("variance_results.json", "w"), indent=1)
print("\nwrote variance_results.json")
