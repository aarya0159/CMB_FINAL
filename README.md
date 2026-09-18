# Finite Map Variance in a Simulated Cosmic Microwave Background

Code and data supporting every number, table and figure in the paper.

Aarya Pandey and Vijaya Geet Sudhams Bathamsetty, Global Indian International School, Singapore.

## Contents

| File | Produces |
|---|---|
| `cmb_simulator.py` | The simulator engine. Unmodified; every script below imports it rather than reimplementing its physics. |
| `validation_trials.py` | Table 2, and the 0.054 per cent worst-case figure across 480 trials. |
| `variance_campaign.py` | Table 1, the measured exponent, and the predicted constants 0.81 and 0.300. Writes `variance_results.json`. |
| `width_campaign.py` | The map-width campaign at 192, 384 and 768 pixels. Writes `width_results.json`. |
| `fit_two_component.py` | The two-component fit (A, B, chi-squared), the finite-map floor, and the floor-versus-width test. Writes `fit_results.json`. |
| `make_figure.py` | Figure 1, both panels. Writes `figure1.png`. |

## Running

Requires Python 3 with NumPy, SciPy and Matplotlib. `pygame` is **not** required: the
scripts stub the graphics import so the simulator's physics can be driven headlessly.

```
python validation_trials.py     # Table 2
python variance_campaign.py     # Table 1          -> variance_results.json
python width_campaign.py        # map-width sweep  -> width_results.json
python fit_two_component.py     # fit + floor test -> fit_results.json
python make_figure.py           # Figure 1         -> figure1.png
```

`fit_two_component.py` needs both JSON files, and `make_figure.py` needs all three,
so run them in the order above. Total runtime is roughly ten minutes, almost all of
it in `width_campaign.py` at the 768 x 480 map size.

## Reproducibility

Every realisation is generated from a deterministic integer seed, so the numbers
below come out identically on any machine.

- **Wave-count campaign**: seed = `20000 + 37k + N` for the kth realisation at wave
  count N. Seven wave counts, 1,900 universes, evaluated on the 384 x 240 grid at a
  fixed instant t = 5.0 s in the standing-wave cycle.
- **Map-width campaign**: seed = `k + 1`, the same seed sequence at all three map
  sizes, with the wave count held at N = 384 and both dimensions doubled at each
  step (192 x 120, 384 x 240, 768 x 480), so each step quadruples the map area.
  400, 300 and 200 universes respectively.
- **Validation**: 480 comparisons, five baseline temperatures from 3,000 K to
  2.72548 K, twelve seeds and eight instants in the cycle.

## What the predicted constants are, and where they come from

Both are computed from the simulator's own amplitude distribution before any fitting,
in `variance_campaign.py`. Wavelengths are log-uniform on [18, 260] pixels; each
amplitude is an envelope `exp(-[ln(lambda/70)/0.75]^2)` times a random factor uniform
on [0.6, 1.4]; the standing-wave factor contributes `cos^2`.

With `q = E[a^4]/E[a^2]^2 = 2.401` by Gaussian quadrature (and 2.401 by
four-million-sample Monte Carlo, as a cross-check):

- naive coefficient `sqrt(r)/2 = 0.806`, where `r = 3q/2 - 1 = 2.601`, obtained by
  treating the sum of squared amplitudes as fixed;
- refined coefficient `A = q/8 = 0.300`, obtained by retaining the normalisation the
  code applies.

## Note on the map-width campaign

The original map-width run used 120, 80 and 30 universes and its seed list was not
preserved; `SEED_RECOVERY_REPORT.md` documents the search that failed to recover it.
`width_campaign.py` regenerates the campaign from the documented seed rule above and
defaults to 400, 300 and 200 universes, whose measured floors agree with the published
values within their uncertainties at every map size. `PLAN_AS_PUBLISHED` in that file
runs the published counts instead.

The wave-count campaign, the two-component fit and the validation trials are
unaffected and reproduce the published values exactly.
