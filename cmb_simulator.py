"""
CMB Anisotropy Explorer
=======================
A live, interactive simulation of Cosmic Microwave Background temperature
anisotropies. The fluctuation field is built from a sum of randomized sine
plane waves whose amplitudes oscillate in time as standing waves -- a toy
analogue of the acoustic (baryon-photon) oscillations that produced the
real CMB pattern.

Controls
--------
  Sliders (drag with mouse):
    Universe Age / Cooling ...... baseline temperature, 3000 K -> 2.725 K
    Perturbation Amplitude ...... dT/T contrast of hot/cold spots

  Keyboard:
    LEFT / RIGHT ... cool / reheat the universe
    UP / DOWN ...... raise / lower perturbation amplitude
    SPACE .......... pause / resume the oscillations
    R .............. regenerate the field with a new random seed
    ESC ............ quit

Requires:  pip install pygame numpy
Run:       python cmb_simulator.py
"""

import math
import sys

import numpy as np
import pygame

# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
WIEN_B = 2.897771955e-3        # Wien displacement constant, m*K
NU_PEAK_PER_K = 5.878925757e10  # Planck-law peak frequency, Hz per Kelvin
T_TODAY = 2.72548               # CMB temperature today (Fixsen 2009), K
T_RECOMBINATION = 3000.0        # temperature at last scattering, K


def _build_age_table():
    """Age of the universe vs scale factor from integrating the Friedmann
    equation (flat LCDM, Planck 2018: H0=67.36, Om=0.3153, radiation
    included). Replaces the matter-era (1+z)^-3/2 approximation, whose
    error peaked at ~20% around z~6."""
    h0 = 67.36e3 / 3.0856775814913673e22            # s^-1
    omega_r = 2.4728e-5 / 0.6736 ** 2 * (1.0 + 0.2271 * 3.046)
    omega_m = 0.3153
    omega_l = 1.0 - omega_m - omega_r
    a = np.logspace(-10, 0, 20000)
    ha = h0 * np.sqrt(omega_r * a ** -4 + omega_m * a ** -3 + omega_l)
    f = 1.0 / (a * ha)
    t_s = np.concatenate(([0.0],
                          np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(a))))
    return a, t_s / 3.1557e7                        # years


_AGE_A, _AGE_YR = _build_age_table()


def age_at_redshift(z):
    """Age of the universe (years) at redshift z, LCDM lookup."""
    return float(np.interp(1.0 / (1.0 + max(z, 0.0)), _AGE_A, _AGE_YR))

# ----------------------------------------------------------------------
# Display layout
# ----------------------------------------------------------------------
WIN_W, WIN_H = 1000, 700
MAP_RECT = pygame.Rect(0, 0, WIN_W, 520)
PANEL_RECT = pygame.Rect(0, 520, WIN_W, WIN_H - 520)

FIELD_W, FIELD_H = 384, 240     # internal simulation grid (upscaled to window)
N_WAVES = 48                    # number of superposed plane waves

BG_COLOR = (16, 16, 22)
PANEL_COLOR = (28, 28, 38)
TEXT_COLOR = (225, 225, 230)
DIM_TEXT = (140, 140, 155)
ACCENT = (255, 170, 60)


# ----------------------------------------------------------------------
# Simulation engine
# ----------------------------------------------------------------------
class CMBField:
    """Sum of random sine plane waves with standing-wave time modulation.

    The spatial basis  sin(k.x + phi)  is precomputed once per seed, so each
    frame only needs one small matrix contraction -- fast enough for 60 fps.
    """

    def __init__(self, width, height, n_waves=N_WAVES, seed=None):
        rng = np.random.default_rng(seed)

        # Wavelengths (in grid pixels), log-uniform, with amplitudes peaked
        # around a preferred scale -- a cartoon of the first acoustic peak.
        lam = np.exp(rng.uniform(np.log(18.0), np.log(260.0), n_waves))
        peak_scale = 70.0
        amp = (np.exp(-((np.log(lam / peak_scale)) / 0.75) ** 2)
               * rng.uniform(0.6, 1.4, n_waves))

        k = 2.0 * np.pi / lam
        theta = rng.uniform(0.0, 2.0 * np.pi, n_waves)
        kx, ky = k * np.cos(theta), k * np.sin(theta)
        phi = rng.uniform(0.0, 2.0 * np.pi, n_waves)     # spatial phase
        psi = rng.uniform(0.0, 2.0 * np.pi, n_waves)     # temporal phase

        # Standing-wave frequency: smaller modes oscillate faster (omega ~ k),
        # scaled so the dominant scale has a ~12 s period on screen.
        omega = (2.0 * np.pi / 12.0) * (peak_scale / lam)

        ygrid, xgrid = np.mgrid[0:height, 0:width].astype(np.float32)
        arg = (kx[:, None, None] * xgrid
               + ky[:, None, None] * ygrid
               + phi[:, None, None])
        self.basis = np.sin(arg, dtype=np.float32)        # (n_waves, H, W)

        self.amp = amp.astype(np.float32)
        self.omega = omega.astype(np.float32)
        self.psi = psi.astype(np.float32)

        # Fixed normalization from the expected RMS (spatial and temporal
        # averages of sin^2 each contribute 1/2), so pulsing stays visible
        # instead of being flattened by per-frame rescaling.
        rms = math.sqrt(float(np.sum(self.amp ** 2)) / 4.0)
        self.norm = 1.0 / (2.8 * rms)

    def evaluate(self, t):
        """Return the fluctuation field at time t, roughly in [-1, 1]."""
        coeff = self.amp * np.cos(self.omega * t + self.psi)
        field = np.tensordot(coeff.astype(np.float32), self.basis, axes=1)
        return field * self.norm


# ----------------------------------------------------------------------
# Colormap: cold (blue) -> neutral -> hot (red), CMB-map style
# ----------------------------------------------------------------------
def build_colormap():
    stops = [
        (0.00, (10, 15, 95)),
        (0.20, (25, 80, 200)),
        (0.42, (120, 175, 230)),
        (0.50, (232, 228, 218)),
        (0.58, (250, 195, 105)),
        (0.80, (225, 85, 30)),
        (1.00, (140, 5, 20)),
    ]
    xs = np.array([s[0] for s in stops]) * 255.0
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(np.arange(256), xs, [s[1][c] for s in stops])
    return lut


# ----------------------------------------------------------------------
# UI slider
# ----------------------------------------------------------------------
class Slider:
    def __init__(self, rect, label, vmin, vmax, value, log=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.log = log
        self.dragging = False
        self.set_value(value)

    # position t in [0,1]  <->  value
    def _to_value(self, t):
        if self.log:
            return self.vmin * (self.vmax / self.vmin) ** t
        return self.vmin + (self.vmax - self.vmin) * t

    def _to_t(self, v):
        if self.log:
            return math.log(v / self.vmin) / math.log(self.vmax / self.vmin)
        return (v - self.vmin) / (self.vmax - self.vmin)

    @property
    def value(self):
        return self._to_value(self.t)

    def set_value(self, v):
        self.t = min(1.0, max(0.0, self._to_t(v)))

    def nudge(self, dt):
        self.t = min(1.0, max(0.0, self.t + dt))

    def handle_event(self, event):
        track = self._track_rect()
        grab = track.inflate(0, 14)
        if event.type == pygame.MOUSEBUTTONDOWN and grab.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN,
                                            pygame.MOUSEMOTION):
            self.t = min(1.0, max(0.0, (event.pos[0] - track.x) / track.w))

    def _track_rect(self):
        return pygame.Rect(self.rect.x, self.rect.y + 26, self.rect.w, 6)

    def draw(self, surf, font, value_text):
        surf.blit(font.render(self.label, True, TEXT_COLOR),
                  (self.rect.x, self.rect.y))
        val = font.render(value_text, True, ACCENT)
        surf.blit(val, (self.rect.right - val.get_width(), self.rect.y))

        track = self._track_rect()
        pygame.draw.rect(surf, (60, 60, 75), track, border_radius=3)
        fill = track.copy()
        fill.w = int(track.w * self.t)
        pygame.draw.rect(surf, (90, 120, 190), fill, border_radius=3)
        hx = track.x + int(track.w * self.t)
        pygame.draw.circle(surf, TEXT_COLOR, (hx, track.centery), 9)
        pygame.draw.circle(surf, (90, 120, 190), (hx, track.centery), 9, 2)


# ----------------------------------------------------------------------
# Telemetry formatting
# ----------------------------------------------------------------------
def fmt_temperature(t_kelvin):
    return f"{t_kelvin:,.1f} K" if t_kelvin >= 10 else f"{t_kelvin:.4f} K"


def fmt_frequency(hz):
    if hz >= 1e12:
        return f"{hz / 1e12:.2f} THz"
    return f"{hz / 1e9:.1f} GHz"


def fmt_wavelength(m):
    if m < 1e-6:
        return f"{m * 1e9:.0f} nm"
    if m < 1e-3:
        return f"{m * 1e6:.2f} um"
    return f"{m * 1e3:.3f} mm"


def fmt_delta_t(kelvin):
    if kelvin >= 1.0:
        return f"{kelvin:.2f} K"
    if kelvin >= 1e-3:
        return f"{kelvin * 1e3:.2f} mK"
    return f"{kelvin * 1e6:.1f} uK"


def fmt_age(years):
    if years < 1e6:
        return f"{years / 1e3:.0f} kyr"
    if years < 1e9:
        return f"{years / 1e6:.1f} Myr"
    return f"{years / 1e9:.2f} Gyr"


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("CMB Anisotropy Explorer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    big_font = pygame.font.SysFont("consolas", 20, bold=True)
    small_font = pygame.font.SysFont("consolas", 13)

    lut = build_colormap()
    field = CMBField(FIELD_W, FIELD_H, seed=None)

    slider_temp = Slider((40, PANEL_RECT.y + 22, 380, 40),
                         "Universe Age / Cooling",
                         T_RECOMBINATION, T_TODAY, T_RECOMBINATION, log=True)
    slider_amp = Slider((40, PANEL_RECT.y + 92, 380, 40),
                        "Density Perturbation Amplitude (dT/T)",
                        1e-4, 0.3, 0.02, log=True)

    sim_time = 0.0
    paused = False
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    field = CMBField(FIELD_W, FIELD_H, seed=None)
            slider_temp.handle_event(event)
            slider_amp.handle_event(event)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            slider_temp.nudge(0.6 * dt)      # cool (slider runs hot -> cold)
        if keys[pygame.K_RIGHT]:
            slider_temp.nudge(-0.6 * dt)
        if keys[pygame.K_UP]:
            slider_amp.nudge(0.6 * dt)
        if keys[pygame.K_DOWN]:
            slider_amp.nudge(-0.6 * dt)

        if not paused:
            sim_time += dt

        t_base = slider_temp.value          # baseline temperature, K
        amp = slider_amp.value              # fractional dT/T

        # ---- simulate ------------------------------------------------
        f = field.evaluate(sim_time)        # ~[-1, 1]

        # ---- physics telemetry ----------------------------------------
        t_map = t_base * (1.0 + amp * f)    # actual temperature map, K
        t_avg = float(t_map.mean())
        dt_rms = float(t_map.std())
        nu_peak = NU_PEAK_PER_K * t_avg     # Planck-law peak frequency
        lam_peak = WIEN_B / t_avg           # Wien peak wavelength
        redshift = t_base / T_TODAY - 1.0
        age_yr = age_at_redshift(redshift)

        # ---- render the map -------------------------------------------
        # Visual gain: real dT/T would be invisible, so contrast tracks the
        # amplitude slider (unity gain at dT/T = 0.02).
        gain = math.sqrt(amp / 0.02)
        v = np.clip(f * gain, -1.0, 1.0)
        idx = ((v + 1.0) * 127.5).astype(np.uint8)
        rgb = lut[idx]                                    # (H, W, 3)
        map_surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        map_surf = pygame.transform.smoothscale(map_surf, MAP_RECT.size)

        screen.fill(BG_COLOR)
        screen.blit(map_surf, MAP_RECT.topleft)

        # ---- control panel --------------------------------------------
        pygame.draw.rect(screen, PANEL_COLOR, PANEL_RECT)
        pygame.draw.line(screen, (70, 70, 90),
                         PANEL_RECT.topleft, PANEL_RECT.topright, 2)

        slider_temp.draw(screen, font, fmt_temperature(t_base))
        slider_amp.draw(screen, font, f"{amp:.2e}")

        # ---- telemetry ------------------------------------------------
        tx = 480
        ty = PANEL_RECT.y + 18
        screen.blit(big_font.render("TELEMETRY", True, ACCENT), (tx, ty))
        rows = [
            ("Average Temperature", fmt_temperature(t_avg)),
            ("Peak Frequency  (Planck)", fmt_frequency(nu_peak)),
            ("Peak Wavelength (Wien)", fmt_wavelength(lam_peak)),
            ("dT (rms anisotropy)", fmt_delta_t(dt_rms)),
            ("Redshift z / Age", f"{redshift:,.0f}  /  {fmt_age(age_yr)}"),
        ]
        for i, (name, value) in enumerate(rows):
            y = ty + 30 + i * 22
            screen.blit(font.render(name, True, DIM_TEXT), (tx, y))
            screen.blit(font.render(value, True, TEXT_COLOR), (tx + 270, y))

        status = "PAUSED" if paused else f"{clock.get_fps():.0f} fps"
        help_text = ("LEFT/RIGHT cool-reheat | UP/DOWN amplitude | "
                     "SPACE pause | R new seed | ESC quit")
        screen.blit(small_font.render(help_text, True, DIM_TEXT),
                    (40, PANEL_RECT.bottom - 24))
        screen.blit(small_font.render(status, True, ACCENT),
                    (WIN_W - 90, PANEL_RECT.bottom - 24))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
