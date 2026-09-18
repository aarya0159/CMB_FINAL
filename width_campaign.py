"""Map-width campaign (Figure 1b).

Holds the wave count fixed at N = 384 and rebuilds the field on maps of
192x120, 384x240 and 768x480 pixels, doubling both dimensions at each step so
that each step quadruples the map area. Measures the seed-to-seed scatter of
the map's root-mean-square anisotropy at each size.

If the scatter contains a finite-map floor, the floor should fall as 1/L when
the map is enlarged, while the A/N term stays fixed because N does not change.
The floor itself is extracted in fit_two_component.py, which needs A from the
wave-count campaign.

Writes width_results.json.

Run:  python variance_campaign.py     (first, for A)
      python width_campaign.py

NOTE ON REALISATION COUNTS
--------------------------
PLAN_AS_PUBLISHED holds the counts reported in the paper (120, 80, 30). At
768x480 thirty realisations leave the measured scatter only marginally above
sqrt(A/N), so the floor extracted at that map size swings wildly with the seeds
drawn: from this seed sequence it comes out 0.009 +- 0.013, which is consistent
with the published 0.032 +- 0.008 only because both error bars are large.
PLAN_HIGH_STATS (400, 300, 200) roughly halves every uncertainty, stabilises the
768-pixel point, and returns floors that agree with the published values at all
three map sizes. It is the default for that reason. Switch with
PLAN = PLAN_AS_PUBLISHED to run the campaign at exactly the published counts.
"""
import sys, types, json, math, time
import numpy as np

pg = types.ModuleType("pygame")
class _Rect:
    def __init__(self, *a):
        if len(a) == 4: self.x, self.y, self.w, self.h = a
pg.Rect = _Rect
sys.modules["pygame"] = pg
import cmb_simulator as sim

N_FIXED = 384      # wave count held constant across all three map sizes
TFIX    = 5.0      # same instant in the cycle as the wave-count campaign

# (width, height, number of independently seeded universes)
PLAN_AS_PUBLISHED = [(192, 120, 120), (384, 240, 80), (768, 480, 30)]
PLAN_HIGH_STATS   = [(192, 120, 400), (384, 240, 300), (768, 480, 200)]
PLAN = PLAN_HIGH_STATS

def seed_for(k):
    """Deterministic seed for the kth realisation. The same seed sequence is
    used at every map size, so the three sizes differ only by the grid the
    field is sampled on."""
    return k + 1

results = {}
for W, H, nseeds in PLAN:
    t0 = time.time()
    stds = np.empty(nseeds)
    for k in range(nseeds):
        fld = sim.CMBField(W, H, n_waves=N_FIXED, seed=seed_for(k))
        stds[k] = float(fld.evaluate(TFIX).std())
        del fld
    frac = stds.std(ddof=1) / stds.mean()
    err = frac / math.sqrt(2 * nseeds - 1)
    results[f"{W}x{H}"] = {"width": W, "height": H, "nseeds": nseeds,
                           "mean_std": float(stds.mean()),
                           "frac_scatter": float(frac), "frac_err": float(err)}
    print(f"{W:4d}x{H:<4d} universes={nseeds:4d}  scatter={frac:.4f} +- {err:.4f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)

print(f"\ntotal universes: {sum(p[2] for p in PLAN)}")
json.dump({"n_waves": N_FIXED, "t_fixed": TFIX, "seed_rule": "k + 1",
           "results": results}, open("width_results.json", "w"), indent=1)
print("wrote width_results.json")
