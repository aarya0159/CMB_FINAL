"""Validation trials for the CMB simulator.

Runs the simulator's own engine (cmb_simulator.py, unmodified) and compares
every telemetry readout against an independent evaluation of the same
physics, computed here from CODATA 2018 constants and, for the age, from an
adaptive quadrature of the Friedmann equation using SciPy.

Only the pygame graphics import is stubbed; no physics code path is touched.
Reports the percentage error of each readout at every trial setting and the
worst case per readout across all trials.
"""
import sys, types, math, json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

# ---- stub pygame (graphics only; CMBField and the age table never use it)
pg = types.ModuleType("pygame")
class _Rect:
    def __init__(self, *a):
        if len(a) == 4: self.x, self.y, self.w, self.h = a
        elif len(a) == 1: self.x, self.y, self.w, self.h = a[0]
pg.Rect = _Rect
sys.modules["pygame"] = pg
import cmb_simulator as sim

# ---------------------------------------------------------------- independent
# CODATA 2018
H   = 6.62607015e-34      # J s      (exact)
KB  = 1.380649e-23        # J K^-1   (exact)
C   = 2.99792458e8        # m s^-1   (exact)
MPC = 3.0856775814913673e22
YR  = 3.1557e7

# Wien roots, solved here rather than taken from the simulator
X_FREQ = brentq(lambda x: 3.0 * (1.0 - math.exp(-x)) - x, 1.0, 5.0)
X_WAVE = brentq(lambda x: 5.0 * (1.0 - math.exp(-x)) - x, 1.0, 8.0)
NU_PER_K_IND = X_FREQ * KB / H          # Hz per kelvin
WIEN_B_IND   = H * C / (X_WAVE * KB)    # m K

# Flat LCDM, Planck 2018, same parameters the simulator uses
H0 = 67.36e3 / MPC
OM_R = 2.4728e-5 / 0.6736 ** 2 * (1.0 + 0.2271 * 3.046)
OM_M = 0.3153
OM_L = 1.0 - OM_M - OM_R

def age_independent(z):
    """Age at redshift z by adaptive quadrature: t = int_0^a da/(a H(a))."""
    a_end = 1.0 / (1.0 + z)
    f = lambda a: 1.0 / (a * H0 * math.sqrt(OM_R*a**-4 + OM_M*a**-3 + OM_L))
    val, _ = quad(f, 0.0, a_end, limit=200)
    return val / YR

def pct(sim_val, ind_val):
    if ind_val == 0.0:
        return 0.0 if sim_val == 0.0 else float("inf")
    return 100.0 * abs(sim_val - ind_val) / abs(ind_val)

# ---------------------------------------------------- Table 2 reference case
W, H_PIX = sim.FIELD_W, sim.FIELD_H
_f   = sim.CMBField(W, H_PIX, seed=1).evaluate(0.0)
_ta  = float((3000.0 * (1.0 + 0.02 * _f)).mean())
_z   = 3000.0 / sim.T_TODAY - 1.0
print("TABLE 2  (3,000 K baseline, seed 1, t = 0 s)")
print(f"{'readout':<22}{'simulator':>16}{'independent':>16}")
print(f"{'Average temperature':<22}{_ta:>14.3f} K{'3000.000 K (set)':>16}")
print(f"{'Peak frequency':<22}{sim.NU_PEAK_PER_K*_ta/1e12:>12.4f} THz"
      f"{NU_PER_K_IND*_ta/1e12:>12.4f} THz")
print(f"{'Peak wavelength':<22}{sim.WIEN_B/_ta*1e9:>13.3f} nm"
      f"{WIEN_B_IND/_ta*1e9:>13.3f} nm")
print(f"{'Redshift z':<22}{_z:>16.3f}{3000.0/2.72548-1.0:>16.3f}")
print(f"{'Age':<22}{sim.age_at_redshift(_z)/1e3:>11.3f} kyr"
      f"{age_independent(3000.0/2.72548-1.0)/1e3:>11.3f} kyr")
print()

# ---------------------------------------------------------------- trials
BASELINES = [3000.0, 1500.0, 300.0, 30.0, sim.T_TODAY]
SEEDS     = range(1, 13)
TIMES     = np.arange(0.0, 24.0, 3.0)
AMP       = 0.02

worst = {k: 0.0 for k in ["avg_T", "nu_peak", "lam_peak", "redshift", "age"]}
where = {k: None for k in worst}
rows  = []

for t_base in BASELINES:
    for s in SEEDS:
        fld = sim.CMBField(W, H_PIX, seed=s)
        for t in TIMES:
            f = fld.evaluate(float(t))
            t_map = t_base * (1.0 + AMP * f)
            t_avg = float(t_map.mean())

            # what the simulator reports
            nu_sim  = sim.NU_PEAK_PER_K * t_avg
            lam_sim = sim.WIEN_B / t_avg
            z_sim   = t_base / sim.T_TODAY - 1.0
            age_sim = sim.age_at_redshift(z_sim)

            # independent evaluation of the same quantities
            nu_ind  = NU_PER_K_IND * t_avg
            lam_ind = WIEN_B_IND / t_avg
            z_ind   = t_base / 2.72548 - 1.0
            age_ind = age_independent(z_ind)

            e = {
                "avg_T":    pct(t_avg, t_base),      # finite-map mean offset
                "nu_peak":  pct(nu_sim, nu_ind),
                "lam_peak": pct(lam_sim, lam_ind),
                "redshift": pct(z_sim, z_ind),
                "age":      pct(age_sim, age_ind),
            }
            for k, v in e.items():
                if v > worst[k]:
                    worst[k] = v
                    where[k] = {"T_base": t_base, "seed": s, "t": float(t)}
            rows.append({"T_base": t_base, "seed": s, "t": float(t), **e})

# ---------------------------------------------------------------- report
print("Independent constants solved here:")
print(f"  x*(frequency form)  = {X_FREQ:.10f}")
print(f"  x*(wavelength form) = {X_WAVE:.10f}")
print(f"  nu_peak per K       = {NU_PER_K_IND:.6e} Hz/K   "
      f"(simulator: {sim.NU_PEAK_PER_K:.6e})")
print(f"  Wien b              = {WIEN_B_IND:.9e} m K     "
      f"(simulator: {sim.WIEN_B:.9e})")
print(f"\nTrials: {len(rows)} "
      f"({len(BASELINES)} baselines x {len(list(SEEDS))} seeds x {len(TIMES)} times)\n")
print(f"{'readout':<12}{'worst-case error':>18}   worst at")
for k in ["avg_T", "nu_peak", "lam_peak", "redshift", "age"]:
    w = where[k]
    at = ("exact at every setting" if w is None else
          f"T={w['T_base']:.5g} K, seed={w['seed']}, t={w['t']:.0f}s")
    print(f"{k:<12}{worst[k]:>17.5f}%   {at}")
print(f"\nWORST CASE ACROSS ALL READOUTS: {max(worst.values()):.3f}%")

json.dump({"worst": worst, "where": where, "n_trials": len(rows),
           "constants": {"x_freq": X_FREQ, "x_wave": X_WAVE,
                         "nu_per_K_independent": NU_PER_K_IND,
                         "wien_b_independent": WIEN_B_IND}},
          open("validation_results.json", "w"), indent=1)
