"""Two-component model fit, finite-map floor, and the map-width test.

Fits

    (sigma_dT / <dT>)^2  =  A / N  +  B

to the wave-count campaign by weighted least squares in the squared-scatter
variable, with uncertainties propagated as 2*sigma*delta_sigma. A is the
amplitude-statistics term predicted independently as q/8; sqrt(B) is the
finite-map floor.

Then takes the map-width campaign at fixed N = 384, subtracts the known A/N
term at each map size, and tests the successive floor ratios against the
factor of 2.00 expected if the floor falls as 1/L.

Reads variance_results.json and width_results.json, writes fit_results.json.

Run:  python fit_two_component.py
"""
import json, math
import numpy as np

V = json.load(open("variance_results.json"))
res = V["results"]
N  = np.array(sorted(int(k) for k in res))
s  = np.array([res[str(n)]["frac_scatter"] for n in N])
ds = np.array([res[str(n)]["frac_err"]     for n in N])

# ---- weighted least squares on y = s^2 against the model A*(1/N) + B ------
y  = s ** 2
dy = 2 * s * ds                       # propagated uncertainty on s^2
w  = 1.0 / dy ** 2
X  = np.column_stack([1.0 / N, np.ones_like(N, dtype=float)])

XtW  = X.T * w
cov  = np.linalg.inv(XtW @ X)
beta = cov @ (XtW @ y)
A, B = beta
dA, dB = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])

model_y = X @ beta
resid   = (y - model_y) / dy
chi2    = float((resid ** 2).sum())
dof     = len(N) - 2

floor   = math.sqrt(B)
dfloor  = dB / (2 * floor)

print("TWO-COMPONENT FIT   (sigma/<dT>)^2 = A/N + B")
print(f"  A      = {A:.4f} +- {dA:.4f}     (predicted q/8 = {V['A_theory']:.4f},"
      f" {abs(A-V['A_theory'])/dA:.1f} sigma)")
print(f"  B      = {B:.6f} +- {dB:.6f}")
print(f"  floor  = sqrt(B) = {floor:.4f} +- {dfloor:.4f}")
print(f"  chi^2  = {chi2:.2f} on {dof} degrees of freedom")
i = int(np.argmax(abs(resid)))
print(f"  largest residual = {abs(resid[i]):.1f} sigma at N = {N[i]}\n")

print(f"{'N':>5}{'measured':>12}{'model':>9}{'residual':>10}")
for n, sv, dsv, my in zip(N, s, ds, model_y):
    print(f"{n:>5}{sv:>9.4f} +-{dsv:.4f}{math.sqrt(my):>9.3f}"
          f"{(sv**2-my)/(2*sv*dsv):>9.1f}s")

# ---- map-width test: strip the A/N term, test the 1/L law ----------------
Wd = json.load(open("width_results.json"))
NW = Wd["n_waves"]
keys = sorted(Wd["results"], key=lambda k: Wd["results"][k]["width"])

print(f"\nMAP-WIDTH TEST at fixed N = {NW}   (floor = sqrt(scatter^2 - A/N))")
floors, dfloors, widths = [], [], []
for k in keys:
    R = Wd["results"][k]
    sv, dsv = R["frac_scatter"], R["frac_err"]
    var = sv ** 2 - A / NW
    if var <= 0:
        print(f"  {k}: scatter {sv:.4f} lies below sqrt(A/N); floor unresolved")
        continue
    f = math.sqrt(var)
    # propagate the scatter and the fitted A
    df = math.sqrt((sv * dsv / f) ** 2 + (dA / (2 * NW * f)) ** 2)
    floors.append(f); dfloors.append(df); widths.append(R["width"])
    print(f"  {k:>9}  n={R['nseeds']:>4}  scatter={sv:.4f} +- {dsv:.4f}"
          f"   floor={f:.4f} +- {df:.4f}")

print("\n  successive ratios (expected 2.00 if floor ~ 1/L):")
ratios = []
for a in range(len(floors) - 1):
    rt = floors[a] / floors[a + 1]
    drt = rt * math.hypot(dfloors[a] / floors[a], dfloors[a + 1] / floors[a + 1])
    ratios.append({"from": widths[a], "to": widths[a + 1],
                   "ratio": rt, "err": drt,
                   "sigma_from_2": abs(2.0 - rt) / drt})
    print(f"    {widths[a]} -> {widths[a+1]}:  {rt:.2f} +- {drt:.2f}"
          f"   ({abs(2.0-rt)/drt:.1f} sigma from 2.00)")

# power-law slope of floor against width
lw, lf = np.log(widths), np.log(floors)
wt = (np.array(floors) / np.array(dfloors)) ** 2
Wm = wt.sum(); xb = (wt*lw).sum()/Wm; yb = (wt*lf).sum()/Wm
sl = (wt*(lw-xb)*(lf-yb)).sum() / (wt*(lw-xb)**2).sum()
sle = 1.0/math.sqrt((wt*(lw-xb)**2).sum())
print(f"\n  floor ~ L^({sl:.2f} +- {sle:.2f})    (expected -1.00)")

json.dump({"A": A, "A_err": dA, "B": B, "B_err": dB,
           "floor": floor, "floor_err": dfloor,
           "chi2": chi2, "dof": dof,
           "largest_residual": {"N": int(N[i]), "sigma": float(abs(resid[i]))},
           "model_values": {str(int(n)): float(math.sqrt(m))
                            for n, m in zip(N, model_y)},
           "width_floors": [{"width": w_, "floor": f_, "err": d_}
                            for w_, f_, d_ in zip(widths, floors, dfloors)],
           "ratios": ratios,
           "floor_slope": {"slope": sl, "err": sle}},
          open("fit_results.json", "w"), indent=1)
print("\nwrote fit_results.json")
