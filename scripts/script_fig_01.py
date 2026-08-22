#!/usr/bin/env python3
"""
Figure 1 -- Completeness-magnitude selection by b-value stability.

Generates `fig_mc_stability,png` for the manuscript
"Fractal and Multifractal Characterisation of Seismicity ...".

The figure is produced by the SAME estimators described in the paper:
  * Aki-Utsu maximum-likelihood b-value      (Eq. 8,  eq:bmle)
  * Shi & Bolt standard error  sigma_b=b/sqrt(n)  (Eq. 9,  eq:bse)
  * b-Mc stability scan        (mc_stability)
  * Gardner-Knopoff declustering windows     (Eq. 14, eq:gkwindows)

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# --------------------------------------------------------------------------
# 0.  Reproducibility + typography (match the Springer Nature serif look)
# --------------------------------------------------------------------------
rng = np.random.default_rng(20250617)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType (editable / not bitmap)
    "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe pairing
C_FULL = "#0072B2"   # blue   -- full catalogue
C_DECL = "#D55E00"   # vermil -- declustered catalogue
C_MC   = "#444444"   # grey   -- Mc guide line

# --------------------------------------------------------------------------
# 1.  Paper estimators (verbatim logic of Listings 5 & 6 / mc_stability)
# --------------------------------------------------------------------------
def b_value(mag, dM=0.1, Mc=None):
    """Aki-Utsu MLE b-value + Shi&Bolt error; max-curvature Mc if Mc=None."""
    mag = np.asarray(mag, float)
    if Mc is None:                                   # maximum-curvature Mc
        br = np.arange(mag.min(), mag.max() + dM, dM)
        counts, edges = np.histogram(mag, bins=br)
        mids = 0.5 * (edges[:-1] + edges[1:])
        Mc = mids[np.argmax(counts)]
    m = mag[mag >= Mc - 1e-9]
    n = m.size
    if n < 2:
        return dict(b=np.nan, Mc=Mc, n=n, se=np.nan)
    b = np.log10(np.e) / (m.mean() - (Mc - dM / 2.0))   # Eq. (8)
    se = b / np.sqrt(n)                                  # Eq. (9)
    return dict(b=b, Mc=Mc, n=n, se=se)


def mc_stability(mag, dM=0.1, n_min=50):
    """b recomputed over increasing cutoffs -> (cuts, b, se, n)."""
    cuts = np.arange(round(mag.min(), 1), mag.max() - 1.0, dM)
    b, se, n = [], [], []
    for mc in cuts:
        r = b_value(mag, dM, Mc=mc)
        b.append(r["b"]); se.append(r["se"]); n.append(r["n"])
    cuts, b, se, n = map(np.asarray, (cuts, b, se, n))
    keep = n >= n_min
    return cuts[keep], b[keep], se[keep], n[keep]


def gk_decluster(t_days, lon, lat, mag):
    """Gardner-Knopoff (1974) space-time window declustering -> keep mask."""
    L = 10.0 ** (0.1238 * mag + 0.983)               # km   (Eq. 14)
    T = np.where(mag < 6.5, 10.0 ** (0.5409 * mag - 0.547),
                            10.0 ** (0.032 * mag + 2.7389))   # days
    aftershock = np.zeros(mag.size, bool)
    deg = 111.19
    for i in np.argsort(mag)[::-1]:                  # mainshocks first
        if aftershock[i]:
            continue
        dt = t_days - t_days[i]
        dx = (lon - lon[i]) * deg * np.cos(np.radians(lat[i]))
        dy = (lat - lat[i]) * deg
        dr = np.hypot(dx, dy)
        hit = (dt > 0) & (dt <= T[i]) & (dr <= L[i]) & (~aftershock)
        aftershock[hit] = True
    return ~aftershock                               # True = independent


# --------------------------------------------------------------------------
# 2.  Simulate a Marmara-like catalogue (only to feed the estimators)
# --------------------------------------------------------------------------
B_TRUE   = 0.95                       # underlying GR b-value
BETA     = B_TRUE * np.log(10.0)
M_MIN    = 1.0                        # lowest simulated magnitude
MU_DET   = 2.25                       # detection midpoint  -> Mc ~ 2.3-2.5
SIG_DET  = 0.12                       # detection roll-off width
# Marmara window from the manuscript: ~40.2-41.3 N, 27.5-30.5 E
LON0, LON1 = 27.5, 30.5
LAT0, LAT1 = 40.2, 41.3
T_YEARS  = 25.0
T_DAYS   = T_YEARS * 365.25

def tapered_mags(n):
    return M_MIN + rng.exponential(1.0 / BETA, size=n)

def detect(mag):
    from scipy.special import erf
    q = 0.5 * (1.0 + erf((mag - MU_DET) / (np.sqrt(2.0) * SIG_DET)))
    return rng.random(mag.size) < q

# --- (a) background seismicity -------------------------------------------
n_bg = 30000
m_bg = tapered_mags(n_bg)
t_bg = rng.uniform(0, T_DAYS, n_bg)
lon_bg = rng.uniform(LON0, LON1, n_bg)
lat_bg = rng.uniform(LAT0, LAT1, n_bg)

# --- (b) aftershock clusters (Omori-like) around larger mainshocks -------
big = m_bg >= 3.4
idx_ms = np.where(big)[0]
rng.shuffle(idx_ms)
idx_ms = idx_ms[:55]                                 # productive mainshocks

m_af, t_af, lon_af, lat_af = [], [], [], []
for i in idx_ms:
    Mm = m_bg[i]
    n_a = int(rng.poisson(10.0 ** (0.8 * (Mm - 2.0)) * 9))   # productivity
    if n_a == 0:
        continue
    am = tapered_mags(n_a)
    am = am[am < Mm]                                  # aftershocks smaller than mainshock
    n_a = am.size
    Lm = 10.0 ** (0.1238 * Mm + 0.983)               # GK distance scale (km)
    Tm = (10.0 ** (0.5409 * Mm - 0.547)) if Mm < 6.5 else 10.0 ** (0.032 * Mm + 2.7389)
    # Omori-decaying times within ~the GK window
    dt = (rng.random(n_a) ** 3) * min(Tm, T_DAYS * 0.6)
    at = np.clip(t_bg[i] + dt, 0, T_DAYS)
    # tight spatial cloud (well inside the GK radius)
    rad_km = np.abs(rng.normal(0, Lm * 0.25, n_a))
    ang = rng.uniform(0, 2 * np.pi, n_a)
    dlat = (rad_km * np.sin(ang)) / 111.19
    dlon = (rad_km * np.cos(ang)) / (111.19 * np.cos(np.radians(lat_bg[i])))
    m_af.append(am); t_af.append(at)
    lon_af.append(np.clip(lon_bg[i] + dlon, LON0, LON1))
    lat_af.append(np.clip(lat_bg[i] + dlat, LAT0, LAT1))

m_af   = np.concatenate(m_af)
t_af   = np.concatenate(t_af)
lon_af = np.concatenate(lon_af)
lat_af = np.concatenate(lat_af)

# --- (c) merge, then apply detection incompleteness ----------------------
mag = np.concatenate([m_bg, m_af])
tt  = np.concatenate([t_bg, t_af])
lon = np.concatenate([lon_bg, lon_af])
lat = np.concatenate([lat_bg, lat_af])

seen = detect(mag)
mag, tt, lon, lat = mag[seen], tt[seen], lon[seen], lat[seen]

# --- (d) Gardner-Knopoff declustering ------------------------------------
keep = gk_decluster(tt, lon, lat, mag)
mag_decl = mag[keep]

print(f"Full catalogue        : {mag.size:6d} events")
print(f"Declustered catalogue : {mag_decl.size:6d} events "
      f"({100*mag_decl.size/mag.size:.1f}% retained)")

# --------------------------------------------------------------------------
# 3.  b-Mc stability scans + Mc selection
# --------------------------------------------------------------------------
cf, bf, sf, nf = mc_stability(mag,      dM=0.1, n_min=50)
cd, bd, sd, nd = mc_stability(mag_decl, dM=0.1, n_min=40)

def pick_mc(cuts, b, se, span=0.5, dM=0.1):
    """b-value-stability Mc (Woessner & Wiemer 2005, MBS): the lowest cutoff at
    which the average b over the forward half-magnitude window [Mc, Mc+span]
    agrees with b(Mc) to within its one-standard-error envelope."""
    w = int(round(span / dM))
    for k in range(len(cuts) - w):
        b_avg = np.nanmean(b[k:k + w + 1])
        if abs(b_avg - b[k]) <= se[k]:
            return cuts[k]
    # fallback: flattest point
    return cuts[np.nanargmin(np.abs(np.gradient(b)))]

Mc_full = pick_mc(cf, bf, sf)
mc_maxc = b_value(mag, dM=0.1, Mc=None)["Mc"]   # maximum-curvature cross-check
b_at_mc = b_value(mag, dM=0.1, Mc=Mc_full)
print(f"Maximum-curvature Mc  : {mc_maxc:.2f}")
print(f"b-stability Mc        : {Mc_full:.2f}")
print(f"b(Mc)                 : {b_at_mc['b']:.3f} +/- {b_at_mc['se']:.3f} "
      f"(n={b_at_mc['n']})")

# --------------------------------------------------------------------------
# 4.  Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.3))

# stable-plateau shading (visual aid, lightly drawn behind everything)
ax.axvspan(Mc_full, cf.max(), color=C_FULL, alpha=0.045, lw=0, zorder=0)

# --- full catalogue ---
ax.fill_between(cf, bf - sf, bf + sf, color=C_FULL, alpha=0.18, lw=0, zorder=1)
ax.plot(cf, bf, "-", color=C_FULL, lw=1.7, zorder=4,
        marker="o", ms=3.0, mfc="white", mec=C_FULL, mew=0.9,
        label="Full catalogue")

# --- declustered catalogue ---
ax.fill_between(cd, bd - sd, bd + sd, color=C_DECL, alpha=0.16, lw=0, zorder=2)
ax.plot(cd, bd, "--", color=C_DECL, lw=1.7, zorder=5,
        marker="s", ms=3.0, mfc="white", mec=C_DECL, mew=0.9,
        label="Declustered (Gardner\u2013Knopoff)")

# --- Mc vertical line ---
ax.axvline(Mc_full, color=C_MC, lw=1.1, ls=(0, (5, 2)), zorder=3)
ax.annotate(rf"$M_c = {Mc_full:.1f}$",
            xy=(Mc_full, 0.965), xycoords=("data", "axes fraction"),
            xytext=(6, 0), textcoords="offset points",
            ha="left", va="top", color=C_MC, fontsize=10.5)

# 'stable range' guide inside the shaded plateau
ax.annotate("stable range", xy=(0.5 * (Mc_full + cf.max()), 0.90),
            xycoords=("data", "axes fraction"), ha="center", va="center",
            fontsize=9.0, style="italic", color=C_FULL, alpha=0.65)

# small tick mark indicating maximum-curvature cross-check
ax.annotate(r"max-curvature $M_c$", xy=(mc_maxc, 0.05),
            xycoords=("data", "axes fraction"),
            xytext=(-7, 0), textcoords="offset points",
            ha="right", va="bottom", fontsize=8.0, color=C_MC, alpha=0.9)
ax.plot([mc_maxc], [0.0], marker="^", ms=5, color=C_MC,
        transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)

# axes cosmetics
ax.set_xlabel(r"Cutoff magnitude $M_{\mathrm{cut}}$")
ax.set_ylabel(r"$b$-value")
ax.set_xlim(cf.min() - 0.05, max(cf.max(), cd.max()) + 0.05)
ax.set_ylim(0.55, 1.30)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

leg = ax.legend(loc="lower right", fontsize=9.8, handlelength=2.4,
                borderaxespad=0.6, labelspacing=0.35)

fig.tight_layout(pad=0.6)
fig.savefig("figures/fig_mc_stability.pdf", bbox_inches="tight")
fig.savefig("figures/fig_mc_stability.png", dpi=220, bbox_inches="tight")
print("saved fig_mc_stability.pdf / .png")
