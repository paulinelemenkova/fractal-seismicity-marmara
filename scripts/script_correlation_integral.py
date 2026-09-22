#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

rng = np.random.default_rng(40231)
LON0, LON1, LAT0, LAT1 = 27.5, 30.5, 40.2, 41.3
KM = 111.19; COS = np.cos(np.radians(0.5 * (LAT0 + LAT1))); MC = 2.4

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# --- regenerate the Fig. 2 epicentre set --------------------------------
north = np.array([(27.52,40.97),(27.90,40.93),(28.30,40.84),(28.72,40.81),(29.10,40.74),
                  (29.45,40.72),(29.80,40.75),(30.10,40.77),(30.48,40.78)])
central = np.array([(27.55,40.52),(28.00,40.53),(28.50,40.49),(29.00,40.46),(29.50,40.44),
                    (30.00,40.41),(30.30,40.39)])
south = np.array([(27.70,40.28),(28.30,40.27),(28.90,40.26),(29.50,40.27),(30.10,40.30)])
strands = [north, central, south]; wgt = [0.60, 0.27, 0.13]

def along(poly, n, sig):
    seg = np.diff(poly, axis=0); L = np.hypot(seg[:,0]*COS, seg[:,1])*KM
    cum = np.concatenate([[0], np.cumsum(L)]); s = rng.uniform(0, cum[-1], n)
    lon = np.interp(s, cum, poly[:,0]); lat = np.interp(s, cum, poly[:,1])
    op = rng.normal(0, sig, n); pa = rng.normal(0, sig*1.6, n)
    k = np.clip(np.searchsorted(cum, s)-1, 0, len(seg)-1); th = np.arctan2(seg[k,1], seg[k,0]*COS)
    dx = pa*np.cos(th)-op*np.sin(th); dy = pa*np.sin(th)+op*np.cos(th)
    return lon+dx/(KM*COS), lat+dy/KM

def grm(n, mc=MC, mmax=6.3):
    m = mc + rng.exponential(1/(0.95*np.log(10)), int(n*1.4)); m = m[m <= mmax]
    while m.size < n:
        e = mc + rng.exponential(1/(0.95*np.log(10)), n); m = np.concatenate([m, e[e <= mmax]])
    return m[:n]

N = 1300; lo=[]; la=[]; nf = int(N*0.93)
for p, wi in zip(strands, wgt):
    ni = int(round(nf*wi)); a, b = along(p, ni, 3.3 if wi > 0.5 else 4.6); lo += [a]; la += [b]; grm(ni)
nb = N - int(round(nf*0.60)) - int(round(nf*0.27)) - int(round(nf*0.13))
lo += [rng.uniform(LON0, LON1, nb)]; la += [rng.uniform(LAT0, LAT1, nb)]
lon = np.concatenate(lo); lat = np.concatenate(la)
ins = (lon>=LON0)&(lon<=LON1)&(lat>=LAT0)&(lat<=LAT1); lon, lat = lon[ins], lat[ins]
x = (lon-LON0)*KM*COS; y = (lat-LAT0)*KM

# --- correlation integral ------------------------------------------------
d = pdist(np.column_stack([x, y]))
r = np.logspace(np.log10(0.5), np.log10(d.max()*1.05), 46)
C = np.array([(d < rr).mean() for rr in r])
ok = C > 0; r, C = r[ok], C[ok]

# --- fit the scaling window ---------------------------------------------
R_LO, R_HI = 5.0, 50.0
win = (r >= R_LO) & (r <= R_HI)
coef, cov = np.polyfit(np.log10(r[win]), np.log10(C[win]), 1, cov=True)
D2, b0 = coef; D2_se = np.sqrt(cov[0, 0])
pred = coef[0]*np.log10(r[win]) + coef[1]
ss_res = np.sum((np.log10(C[win]) - pred)**2)
ss_tot = np.sum((np.log10(C[win]) - np.log10(C[win]).mean())**2)
R2 = 1 - ss_res/ss_tot
print(f"N={lon.size}  D2={D2:.2f}+/-{D2_se:.2f}  R2={R2:.3f}  window=[{R_LO:.0f},{R_HI:.0f}] km  npts={win.sum()}")

# --- plot ----------------------------------------------------------------
C_IN = "#1f6fb2"; C_OUT = "#9aa0a6"; C_FIT = "#c0392b"
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.set_xscale("log"); ax.set_yscale("log")

ax.axvspan(R_LO, R_HI, color=C_IN, alpha=0.07, lw=0, zorder=0)
ax.plot(r, C, "-", color="#c9ced3", lw=1.0, zorder=1)
ax.plot(r[~win], C[~win], "o", ms=4.2, mfc="white", mec=C_OUT, mew=1.0, zorder=3)
ax.plot(r[win], C[win], "o", ms=4.6, mfc=C_IN, mec="white", mew=0.5, zorder=4,
        label="scaling range (fitted)")

# fitted D2 line (extended a little beyond the window)
rr = np.logspace(np.log10(R_LO*0.7), np.log10(R_HI*1.25), 50)
ax.plot(rr, 10**(b0)*rr**D2, "--", color=C_FIT, lw=1.8, zorder=5,
        label=rf"fit: $D_2={D2:.2f}\pm{D2_se:.2f}$")

# slope-2 (embedding-dimension) reference guide in the small-r roll-off
rg = np.logspace(np.log10(0.7), np.log10(3.5), 20)
Cg = C[np.argmin(np.abs(r-1.5))] * (rg/1.5)**2
ax.plot(rg, Cg, ":", color="#555", lw=1.3, zorder=3)
ax.text(1.05, Cg[np.argmin(np.abs(rg-1.05))]*0.55, "slope 2\n(embedding dim.)",
        fontsize=7.6, color="#555", ha="center", va="top", style="italic")

# annotations for the excluded roll-offs
ax.annotate("location error /\ncatalogue resolution", xy=(2.0, C[np.argmin(np.abs(r-2.0))]),
            xytext=(2.1, 5e-4), fontsize=8.0, color="#444", ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
ax.annotate("finite-area\nsaturation", xy=(150, C[np.argmin(np.abs(r-150))]),
            xytext=(150, 6e-2), fontsize=8.0, color="#444", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))

# scaling-window readout
ax.text(0.035, 0.95,
        rf"scaling window: $r\in[{R_LO:.0f},\,{R_HI:.0f}]$ km"
        f"\n"
        rf"$D_2 = {D2:.2f}\pm{D2_se:.2f}$   ($R^2={R2:.3f}$)",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", lw=0.7, alpha=0.92))

ax.set_xlabel(r"Inter-event separation $r$ (km)")
ax.set_ylabel(r"Correlation integral $C(r)$")
ax.set_xlim(0.5, 300); ax.set_ylim(5e-5, 1.5)
ax.grid(True, which="major", color="#dddddd", lw=0.5, zorder=0)
ax.grid(True, which="minor", color="#eeeeee", lw=0.4, zorder=0)
ax.legend(loc="lower right", fontsize=9.0, frameon=True, framealpha=0.92,
          edgecolor="#bbb", handlelength=2.2)

fig.tight_layout(pad=0.5)
fig.savefig("figures/fig_correlation_integral.pdf", bbox_inches="tight")
fig.savefig("figures/fig_correlation_integral.png", dpi=230, bbox_inches="tight")
print("saved")
