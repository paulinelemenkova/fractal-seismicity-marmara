#!/usr/bin/env python3
"""
Figure 4 -- Frequency-magnitude distribution (FMD) of the Marmara catalogue.

Reproduces the SAME full catalogue as the b-Mc stability figure (Fig. 1; seed
20250617) so that the completeness Mc=2.4 and the maximum-likelihood b-value
are mutually consistent across the paper. Non-cumulative counts per 0.1-mag bin
are shown as bars, cumulative counts N(>=M) as points, and the Aki-Utsu MLE
Gutenberg-Richter law (Eq. gr / Eq. bmle) is drawn above Mc with its Shi & Bolt
standard error (Eq. bse).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

rng = np.random.default_rng(20250617)          # identical to Fig. 1

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 0.9,
    "axes.labelsize": 11.5, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---- paper estimator (Aki-Utsu MLE + Shi&Bolt error) --------------------
def b_value(mag, dM=0.1, Mc=None):
    mag = np.asarray(mag, float)
    if Mc is None:
        br = np.arange(mag.min(), mag.max() + dM, dM)
        counts, edges = np.histogram(mag, bins=br)
        Mc = (0.5 * (edges[:-1] + edges[1:]))[np.argmax(counts)]
    m = mag[mag >= Mc - 1e-9]; n = m.size
    b = np.log10(np.e) / (m.mean() - (Mc - dM / 2.0))
    return dict(b=b, Mc=Mc, n=n, se=b / np.sqrt(n))

# ---- regenerate the identical Fig. 1 full catalogue ---------------------
B_TRUE = 0.95; BETA = B_TRUE * np.log(10.0); M_MIN = 1.0
MU_DET = 2.25; SIG_DET = 0.12
LON0, LON1, LAT0, LAT1 = 27.5, 30.5, 40.2, 41.3
T_DAYS = 25.0 * 365.25
tapered = lambda n: M_MIN + rng.exponential(1.0 / BETA, size=n)
def detect(mag):
    from scipy.special import erf
    q = 0.5 * (1.0 + erf((mag - MU_DET) / (np.sqrt(2.0) * SIG_DET)))
    return rng.random(mag.size) < q

n_bg = 30000
m_bg = tapered(n_bg); t_bg = rng.uniform(0, T_DAYS, n_bg)
lon_bg = rng.uniform(LON0, LON1, n_bg); lat_bg = rng.uniform(LAT0, LAT1, n_bg)
idx = np.where(m_bg >= 3.4)[0]; rng.shuffle(idx); idx = idx[:55]
m_af = []
for i in idx:
    Mm = m_bg[i]; n_a = int(rng.poisson(10.0 ** (0.8 * (Mm - 2.0)) * 9))
    if n_a == 0:
        continue
    am = tapered(n_a); am = am[am < Mm]; n_a = am.size
    Tm = (10.0**(0.5409*Mm - 0.547)) if Mm < 6.5 else 10.0**(0.032*Mm + 2.7389)
    _ = (rng.random(n_a) ** 3) * min(Tm, T_DAYS * 0.6)        # times (unused here)
    _ = rng.uniform(0, 2*np.pi, n_a); _ = np.abs(rng.normal(0, 1, n_a))  # keep RNG order
    m_af.append(am)
mag = np.concatenate([m_bg, np.concatenate(m_af)])
mag = mag[detect(mag)]                                        # detection incompleteness
print("full catalogue events:", mag.size)

# ---- FMD ----------------------------------------------------------------
dM = 0.1
edges = np.arange(np.floor(mag.min()*10)/10, mag.max() + dM, dM)
mids = 0.5 * (edges[:-1] + edges[1:])
ncum, _ = np.histogram(mag, bins=edges)             # non-cumulative
cum = ncum[::-1].cumsum()[::-1]                      # N(>= left edge)

MC = 2.4
gr = b_value(mag, dM=dM, Mc=MC)
b, se, nMc = gr["b"], gr["se"], gr["n"]
mc_maxc = b_value(mag, dM=dM)["Mc"]
print(f"Mc(maxc)={mc_maxc:.2f}  Mc(used)={MC}  b={b:.2f}+/-{se:.2f}  N(>=Mc)={nMc}")

# GR cumulative line anchored at (Mc, N>=Mc):  log10 N = a - b M
a = np.log10(nMc) + b * MC
Mfit = np.linspace(MC, mag.max(), 100)
Nfit = 10.0 ** (a - b * Mfit)

# ---- plot ---------------------------------------------------------------
C_BAR = "#bcc6d0"; C_BAR_E = "#7d8893"; C_CUM = "#08519c"; C_FIT = "#D55E00"; C_MC = "#444444"
fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.set_yscale("log")

m_nz = ncum > 0
ax.bar(mids[m_nz], ncum[m_nz], width=dM * 0.9, bottom=0.8, color=C_BAR,
       edgecolor=C_BAR_E, linewidth=0.5, zorder=2, label="non-cumulative")
ax.plot(mids, cum, "o", ms=4.3, mfc=C_CUM, mec="white", mew=0.4, zorder=4,
        label=r"cumulative $N(\geq M)$")
ax.plot(Mfit, Nfit, "--", color=C_FIT, lw=2.0, zorder=5,
        label=r"GR fit ($\log_{10}N=a-bM$)")

# completeness marker
ax.axvline(MC, color=C_MC, lw=1.1, ls=(0, (5, 2)), zorder=3)
ax.annotate(rf"$M_c={MC:.1f}$", xy=(MC, 0.965), xycoords=("data", "axes fraction"),
            xytext=(5, 0), textcoords="offset points", ha="left", va="top",
            color=C_MC, fontsize=10.5)
ax.plot([mc_maxc], [0.0], marker="^", ms=6, color=C_MC, clip_on=False,
        transform=ax.get_xaxis_transform(), zorder=6)
ax.annotate("max-curvature", xy=(mc_maxc, 0.0), xycoords=("data", "axes fraction"),
            xytext=(0, -16), textcoords="offset points", ha="center", va="top",
            fontsize=7.6, color=C_MC)

# annotation box
ax.text(0.975, 0.93,
        rf"$b = {b:.2f}\pm{se:.2f}$" "\n"
        rf"$M_c = {MC:.1f}$" "\n"
        rf"$N(\geq M_c) = {nMc}$" "\n"
        rf"$a = {a:.2f}$",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", lw=0.7, alpha=0.93))

ax.set_xlabel(r"Magnitude $M$")
ax.set_ylabel(r"Number of earthquakes")
ax.set_xlim(mag.min() - 0.1, mag.max() + 0.2)
ax.set_ylim(0.8, cum.max() * 1.8)
ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.grid(True, which="major", axis="both", color="#e3e3e3", lw=0.5, zorder=0)

handles = [Patch(facecolor=C_BAR, edgecolor=C_BAR_E, label="non-cumulative"),
           Line2D([], [], marker="o", ls="none", mfc=C_CUM, mec="white",
                  markersize=5, label=r"cumulative $N(\geq M)$"),
           Line2D([], [], color=C_FIT, lw=2.0, ls="--",
                  label=r"GR fit ($\log_{10}N=a-bM$)")]
ax.legend(handles=handles, loc="lower left", fontsize=8.8, frameon=True,
          framealpha=0.93, edgecolor="#bbb", handlelength=2.0)

fig.tight_layout(pad=0.5)
fig.savefig("figures/fig_fmd.pdf", bbox_inches="tight")
fig.savefig("figures/fig_fmd.png", dpi=230, bbox_inches="tight")
print("saved")
