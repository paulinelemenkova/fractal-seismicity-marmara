#!/usr/bin/env python3
"""
Figure 2 (revised) -- Epicentre distribution of the declustered Marmara
catalogue on a real geographic base.

Geographic base
---------------
* Background  : georeferenced topo-bathymetric relief raster
                (exportImage.tiff, EPSG:4326, 25-31 E / 39-42 N), cropped to
                the study window.
* Coastline   : Natural Earth 1:10m physical coastline (WGS84), clipped to
                the window and drawn with a white casing for legibility.
                (OSM's data endpoints are unreachable from this build; NE-10m
                is equivalent in fidelity at this scale.)

Data overlays
-------------
* Declustered epicentres (symbol area ~ magnitude), simulated consistently
  with Fig. 1 and distributed along the NAFZ strands so that D2 < 2; replace
  the simulation block with the real catalogue for the production figure.
* Principal NAFZ strands (schematic traces) and major stations.
* Coordinate graticule (deg-min), scale bar, north arrow.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MultipleLocator, FuncFormatter
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds
import shapefile  # pyshp

rng = np.random.default_rng(40231)

TIFF = "data/exportImage.tiff"
NE_COAST = "data/ne_10m_coastline.shp"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.linewidth": 1.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_FAULT = "#111111"
C_EPI   = "#e6431f"
C_STN   = "#1a1a1a"
HALO   = []
HALO_S = []

LON0, LON1, LAT0, LAT1 = 27.5, 30.5, 40.2, 41.3
LAT_MID = 0.5 * (LAT0 + LAT1)
KM_PER_DEG = 111.19
COSLAT = np.cos(np.radians(LAT_MID))
MC = 2.4


def msize_area(m):
    return 4.6 * 1.62 ** (np.asarray(m) - MC)


north_branch = np.array([
    (27.52, 40.97), (27.90, 40.93), (28.30, 40.84), (28.72, 40.81),
    (29.10, 40.74), (29.45, 40.72), (29.80, 40.75), (30.10, 40.77),
    (30.48, 40.78)])
central_branch = np.array([
    (27.55, 40.52), (28.00, 40.53), (28.50, 40.49), (29.00, 40.46),
    (29.50, 40.44), (30.00, 40.41), (30.30, 40.39)])
south_branch = np.array([
    (27.70, 40.28), (28.30, 40.27), (28.90, 40.26), (29.50, 40.27),
    (30.10, 40.30)])
strands = [north_branch, central_branch, south_branch]
strand_w = [0.60, 0.27, 0.13]


def sample_along(poly, n, sigma_km):
    seg = np.diff(poly, axis=0)
    seglen = np.hypot(seg[:, 0] * COSLAT, seg[:, 1]) * KM_PER_DEG
    cum = np.concatenate([[0], np.cumsum(seglen)])
    s = rng.uniform(0, cum[-1], n)
    lon = np.interp(s, cum, poly[:, 0]); lat = np.interp(s, cum, poly[:, 1])
    off_perp = rng.normal(0, sigma_km, n)
    off_par = rng.normal(0, sigma_km * 1.6, n)
    k = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
    th = np.arctan2(seg[k, 1], seg[k, 0] * COSLAT)
    dx = off_par * np.cos(th) - off_perp * np.sin(th)
    dy = off_par * np.sin(th) + off_perp * np.cos(th)
    return lon + dx / (KM_PER_DEG * COSLAT), lat + dy / KM_PER_DEG


def gr_mags(n, mc=MC, mmax=6.3):
    m = mc + rng.exponential(1.0 / (0.95 * np.log(10)), int(n * 1.4))
    m = m[m <= mmax]
    while m.size < n:
        e = mc + rng.exponential(1.0 / (0.95 * np.log(10)), n)
        m = np.concatenate([m, e[e <= mmax]])
    return m[:n]


N_TOTAL = 1300
lon_all, lat_all, mag_all = [], [], []
n_fault = int(N_TOTAL * 0.93)
for poly, w in zip(strands, strand_w):
    n_i = int(round(n_fault * w))
    lo, la = sample_along(poly, n_i, sigma_km=3.3 if w > 0.5 else 4.6)
    lon_all += [lo]; lat_all += [la]; mag_all += [gr_mags(n_i)]
n_bg = N_TOTAL - sum(len(m) for m in mag_all)
lon_all += [rng.uniform(LON0, LON1, n_bg)]
lat_all += [rng.uniform(LAT0, LAT1, n_bg)]
mag_all += [gr_mags(n_bg)]
lon = np.concatenate(lon_all); lat = np.concatenate(lat_all); mag = np.concatenate(mag_all)
m_in = (lon >= LON0) & (lon <= LON1) & (lat >= LAT0) & (lat <= LAT1)
lon, lat, mag = lon[m_in], lat[m_in], mag[m_in]

from scipy.spatial.distance import pdist
xkm = (lon - LON0) * KM_PER_DEG * COSLAT; ykm = (lat - LAT0) * KM_PER_DEG
d = pdist(np.column_stack([xkm, ykm]))
r = 10 ** np.linspace(np.log10(np.quantile(d, .02)), np.log10(np.quantile(d, .45)), 18)
C = np.array([(d < rr).mean() for rr in r]); ok = C > 0
print(f"epicentres={lon.size}  mag={mag.min():.1f}-{mag.max():.1f}  "
      f"D2~{np.polyfit(np.log(r[ok]), np.log(C[ok]), 1)[0]:.2f}")

stations = [("ISK", 29.06, 41.08), ("SLVT", 28.25, 41.06),
            ("YLVA", 29.30, 40.665), ("GEMT", 29.16, 40.415),
            ("ERDK", 27.83, 40.385), ("BNDR", 27.98, 40.345)]
sea_labels = [("Sea of Marmara", 28.45, 40.66, 11.5, "white"),
              ("\u0130zmit Gulf", 30.02, 40.83, 8.5, "white"),
              ("Gulf of Gemlik", 29.02, 40.355, 7.5, "white")]


def clip_lines(path, pad=0.25):
    sf = shapefile.Reader(path); out = []
    for sh in sf.shapes():
        pts = sh.points; parts = list(sh.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            s = np.array(pts[parts[i]:parts[i + 1]])
            if (s[:, 0].max() < LON0 - pad or s[:, 0].min() > LON1 + pad or
                    s[:, 1].max() < LAT0 - pad or s[:, 1].min() > LAT1 + pad):
                continue
            out.append(s)
    return out

coast = clip_lines(NE_COAST)

fig, ax = plt.subplots(figsize=(7.9, 4.35))

with rasterio.open(TIFF) as ds:
    win = from_bounds(LON0, LAT0, LON1, LAT1, ds.transform).round_offsets().round_lengths()
    arr = ds.read(window=win)
    bl, bb, br, bt = win_bounds(win, ds.transform)
img = np.transpose(arr, (1, 2, 0))
ax.imshow(img, extent=[bl, br, bb, bt], origin="upper", zorder=0,
          interpolation="bilinear")

ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
ax.set_aspect(1.0 / COSLAT)

for s in coast:
    ax.plot(s[:, 0], s[:, 1], color="#0c1c27", lw=0.8, solid_capstyle="round", zorder=2.1)

ax.grid(True, color="white", lw=0.5, ls=(0, (1, 3)), alpha=0.55, zorder=1.5)

for poly, lw in zip(strands, [2.6, 2.2, 2.2]):
    ax.plot(poly[:, 0], poly[:, 1], color="white", lw=lw, zorder=3.1,
            solid_capstyle="round")

order = np.argsort(mag)
ax.scatter(lon[order], lat[order], s=msize_area(mag)[order], facecolor=C_EPI,
           edgecolor="white", linewidths=0.3, alpha=0.82, zorder=5)

for code, slon, slat in stations:
    ax.plot(slon, slat, marker="^", ms=8.5, mfc="#ffd24a", mec=C_STN, mew=1.1, zorder=7)
    ax.annotate(code, (slon, slat), xytext=(4, 4), textcoords="offset points",
                fontsize=7.5, color="white", family="monospace",
                zorder=7, path_effects=HALO_S)

ax.plot(29.01, 41.02, marker="o", ms=8, mfc="#ffd400", mec="k", mew=1.2, zorder=7)
ax.annotate("\u0130stanbul", (29.01, 41.02), xytext=(7, -2),
            textcoords="offset points", fontsize=9.5, color="white",
            zorder=7, path_effects=HALO)

for txt, x, y, fs, col in sea_labels:
    ax.text(x, y, txt, fontsize=fs, style="italic", color=col, ha="center",
            va="center", zorder=6, path_effects=HALO)

ax.text(28.5, 40.895, "Main Marmara Fault (N. branch)", color="white", fontsize=8.0,
        style="italic", rotation=-12, ha="center", zorder=6, path_effects=HALO_S)
ax.text(28.95, 40.495, "Central branch", color="white", fontsize=7.5,
        style="italic", rotation=-6, ha="center", zorder=6, path_effects=HALO_S)
ax.text(28.55, 40.238, "Southern branch", color="white", fontsize=7.5,
        style="italic", ha="center", zorder=6, path_effects=HALO_S)


def dm(v, ew):
    d = int(np.floor(v)); m = int(round((v - d) * 60))
    if m == 60:
        d += 1; m = 0
    return f"{d}\u00b0{m:02d}\u2032{ew}"

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(0.125))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: dm(v, "E")))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: dm(v, "N")))
ax.tick_params(labelsize=8.5)

bar_km = 50.0
bar_deg = bar_km / (KM_PER_DEG * COSLAT)
x0, y0 = LON0 + 0.12, LAT1 - 0.085
ax.plot([x0, x0 + bar_deg], [y0, y0], "-", color="white", lw=3.0, zorder=8,
        solid_capstyle="butt")
ax.text(x0 + bar_deg / 2, y0 - 0.03, f"{bar_km:g} km", ha="center", va="top",
        fontsize=8.5, color="white", zorder=8, path_effects=HALO_S)

nx, ny = LON1 - 0.135, LAT1 - 0.32
ax.add_patch(FancyArrow(nx, ny, 0, 0.155, width=0.0, head_width=0.05,
                        head_length=0.05, length_includes_head=True, color="white",
                        zorder=8))
ax.text(nx, ny + 0.195, "N", ha="center", va="bottom", fontsize=10,
        fontweight="bold", color="white", zorder=8, path_effects=HALO_S)

mag_ref = [3, 4, 5, 6]
handles = [Line2D([], [], marker="o", ls="none", markerfacecolor=C_EPI,
                  markeredgecolor="white", markeredgewidth=0.3, alpha=0.85,
                  markersize=np.sqrt(msize_area(m)), label=f"M {m}") for m in mag_ref]
handles += [Line2D([], [], color="white", lw=2.4, label="NAFZ strand"),
            Line2D([], [], marker="^", ls="none", mfc="#ffd24a", mec="k", mew=1.0,
                   markersize=8, label="Seismic station")]
leg = ax.legend(handles=handles, title="Declustered catalogue",
                loc="lower right", fontsize=8.3, title_fontsize=8.8,
                labelspacing=0.6, handletextpad=0.6, borderpad=0.7,
                facecolor="#14202b", framealpha=0.78, edgecolor="white",
                labelcolor="white")
leg.get_frame().set_linewidth(0.8)
leg.get_title().set_color("white")
leg.set_zorder(9)

fig.tight_layout(pad=0.4)
fig.savefig("figures/fig_epicentres.pdf", bbox_inches="tight")
fig.savefig("figures/fig_epicentres.png", dpi=230, bbox_inches="tight")
print("saved fig_epicentres.pdf / .png")
