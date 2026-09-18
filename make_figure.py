"""Builds Figure 1 from the campaign outputs.

(a) seed-to-seed scatter against wave count, with the failed 0.81 N^(-1/2)
    prediction, the fitted two-component model, and the finite-map floor
(b) the floor against map width at fixed N = 384, with the 1/L line

Series are distinguished by line style and marker as well as colour, and the
legend is always present, so identity never depends on colour alone.

Reads variance_results.json, width_results.json and fit_results.json.
Writes figure1.png.

Run:  python make_figure.py
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED, GREY = "#2c4d76", "#b3452e", "#9a9a9a"
INK, MUTED, SURFACE = "#22252a", "#6b7078", "#fcfcfb"

V = json.load(open("variance_results.json"))
Wd = json.load(open("width_results.json"))
F = json.load(open("fit_results.json"))

N  = np.array(sorted(int(k) for k in V["results"]))
s  = np.array([V["results"][str(n)]["frac_scatter"] for n in N])
ds = np.array([V["results"][str(n)]["frac_err"] for n in N])
A, B = F["A"], F["B"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.3), facecolor=SURFACE)

# ---------------------------------------------------------------- panel (a)
grid = np.logspace(math.log10(N.min() * 0.7), math.log10(N.max() * 1.5), 300)
ax.plot(grid, V["naive_coeff"] * grid ** -0.5, ls="--", lw=2, color=RED,
        label=r"predicted $0.81\,N^{-1/2}$")
ax.plot(grid, np.sqrt(A / grid + B), ls="-", lw=2, color=BLUE,
        label=r"model $\sqrt{A/N+B}$")
ax.axhline(math.sqrt(B), ls=":", lw=1.5, color=GREY)
ax.errorbar(N, s, yerr=ds, fmt="o", ms=8, color=BLUE, ecolor=BLUE,
            elinewidth=1.5, capsize=0, zorder=5, label="measured")
ax.annotate(r"finite-map floor $\sqrt{B}$", (N.min() * 0.75, math.sqrt(B)),
            textcoords="offset points", xytext=(2, -14), color=MUTED, fontsize=10)
ax.set(xscale="log", yscale="log", xlabel="Number of waves  $N$",
       ylabel=r"Seed-to-seed scatter  $\sigma_{\Delta T}/\langle\Delta T\rangle$")
ax.set_title("(a)  Scaling with number of waves", color=INK, fontsize=12)

# ---------------------------------------------------------------- panel (b)
wf = F["width_floors"]
L  = np.array([d["width"] for d in wf], dtype=float)
fl = np.array([d["floor"] for d in wf])
dfl = np.array([d["err"] for d in wf])
lg = np.logspace(math.log10(L.min() * 0.85), math.log10(L.max() * 1.18), 100)
bx.plot(lg, fl[0] * L[0] / lg, ls="--", lw=2, color=RED,
        label=r"$1/L$, normalised at $L=%d$" % L[0])
bx.errorbar(L, fl, yerr=dfl, fmt="o", ms=8, color=BLUE, ecolor=BLUE,
            elinewidth=1.5, capsize=0, zorder=5,
            label="measured  ($N=%d$)" % Wd["n_waves"])
bx.set(xscale="log", yscale="log", xlabel="Map width  $L$  (pixels)",
       ylabel=r"Floor  $\sqrt{B}$")
bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bx.set_xticks(L); bx.set_xticklabels([f"{int(v)}" for v in L])
_yt = [0.03, 0.05, 0.10]
bx.set_yticks(_yt); bx.set_yticklabels([f"{v:.2f}" for v in _yt])
bx.set_title("(b)  The floor shrinks with map size", color=INK, fontsize=12)

for p in (ax, bx):
    p.set_facecolor(SURFACE)
    p.grid(True, which="major", color="#e8e8e6", lw=0.8)
    p.tick_params(colors=MUTED, which="both")
    for sp in p.spines.values(): sp.set_color("#c9c9c6")
    p.xaxis.label.set_color(INK); p.yaxis.label.set_color(INK)
    p.legend(frameon=False, fontsize=10, labelcolor=INK)

fig.tight_layout()
fig.savefig("figure1.png", dpi=200, facecolor=SURFACE)
print("wrote figure1.png")
