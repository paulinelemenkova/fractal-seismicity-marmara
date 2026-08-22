#!/usr/bin/env python3
"""
Figure 5 -- Spatial variation of the Gutenberg-Richter b-value along the
Marmara segment of the NAFZ.

Epicentre POSITIONS are identical to the epicentre map (Fig. 2, seed 40231).
Magnitudes are (re)drawn from a spatially varying b-field -- a low-b zone over
the central (locked) Marmara segment and a higher-b zone over the post-1999
Izmit rupture -- and that pattern is then RECOVERED by maximum-likelihood
(Aki-Utsu, Eq. bmle) b estimation in overlapping circular nodes that each
satisfy a minimum-sample threshold; nodes below threshold are left blank.
Real GeoTIFF relief + Natural Earth coastline provide the geographic base.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree
import rasterio
from rasterio.windows import from_bounds, bounds as win_bounds
import shapefile

TIFF = "data/exportImage.tiff"
NE_COAST = "data/ne_10m_coastline.shp"
LON0, LON1, LAT0, LAT1 = 27.5, 30.5, 40.2, 41.3
KM = 111.19; COS = np.cos(np.radians(0.5*(LAT0+LAT1))); MC = 2.4; dM = 0.1

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 1.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

# --- fault strands (same as Fig. 2) --------------------------------------
north = np.array([(27.52,40.97),(27.90,40.93),(28.30,40.84),(28.72,40.81),(29.10,40.74),
                  (29.45,40.72),(29.80,40.75),(30.10,40.77),(30.48,40.78)])
central = np.array([(27.55,40.52),(28.00,40.53),(28.50,40.49),(29.00,40.46),(29.50,40.44),
                    (30.00,40.41),(30.30,40.39)])
south = np.array([(27.70,40.28),(28.30,40.27),(28.90,40.26),(29.50,40.27),(30.10,40.30)])
strands = [north, central, south]; wgt = [0.60, 0.27, 0.13]

# --- regenerate IDENTICAL Fig. 2 positions (consume RNG exactly) ---------
rng = np.random.default_rng(40231)
def along(poly, n, sig):
    seg = np.diff(poly, axis=0); L = np.hypot(seg[:,0]*COS, seg[:,1])*KM
    cum = np.concatenate([[0], np.cumsum(L)]); s = rng.uniform(0, cum[-1], n)
    lon = np.interp(s, cum, poly[:,0]); lat = np.interp(s, cum, poly[:,1])
    op = rng.normal(0, sig, n); pa = rng.normal(0, sig*1.6, n)
    k = np.clip(np.searchsorted(cum, s)-1, 0, len(seg)-1); th = np.arctan2(seg[k,1], seg[k,0]*COS)
    dx = pa*np.cos(th)-op*np.sin(th); dy = pa*np.sin(th)+op*np.cos(th)
    return lon+dx/(KM*COS), lat+dy/KM
def grm_consume(n, mmax=6.3):                 # consume RNG exactly as Fig. 2
    m = MC + rng.exponential(1/(0.95*np.log(10)), int(n*1.4)); m = m[m <= mmax]
    while m.size < n:
        e = MC + rng.exponential(1/(0.95*np.log(10)), n); m = np.concatenate([m, e[e <= mmax]])
    return m[:n]
N = 1300; lo=[]; la=[]; nf = int(N*0.93)
for p, wi in zip(strands, wgt):
    ni = int(round(nf*wi)); a, b = along(p, ni, 3.3 if wi > 0.5 else 4.6); lo += [a]; la += [b]; grm_consume(ni)
nb = N - int(round(nf*0.60)) - int(round(nf*0.27)) - int(round(nf*0.13))
lo += [rng.uniform(LON0, LON1, nb)]; la += [rng.uniform(LAT0, LAT1, nb)]; grm_consume(nb)
lon = np.concatenate(lo); lat = np.concatenate(la)
ins = (lon>=LON0)&(lon<=LON1)&(lat>=LAT0)&(lat<=LAT1); lon, lat = lon[ins], lat[ins]

# --- impose a spatial b-field and (re)draw magnitudes --------------------
def b_field(lon, lat):
    b = 0.95 * np.ones_like(lon)
    b -= 0.30*np.exp(-(((lon-28.50)/0.45)**2 + ((lat-40.83)/0.10)**2))   # locked gap (low b)
    b -= 0.14*np.exp(-(((lon-29.20)/0.22)**2 + ((lat-40.73)/0.08)**2))   # Cinarcik (low b)
    b += 0.20*np.exp(-(((lon-30.05)/0.30)**2 + ((lat-40.78)/0.10)**2))   # post-1999 Izmit (high b)
    return np.clip(b, 0.60, 1.25)
rng2 = np.random.default_rng(777)
b_true_loc = b_field(lon, lat)
mag = MC + rng2.exponential(1.0/(b_true_loc*np.log(10)))
mag = np.minimum(mag, 6.3)
print(f"events={lon.size}  mean imposed b={b_true_loc.mean():.2f}  mag {mag.min():.1f}-{mag.max():.1f}")

# --- gridded MLE b-map (overlapping nodes) -------------------------------
x = (lon-LON0)*KM*COS; y = (lat-LAT0)*KM
tree = cKDTree(np.column_stack([x, y]))
R_KM = 13.0; N_MIN = 50
glon = np.arange(LON0, LON1 + 1e-9, 0.02)
glat = np.arange(LAT0, LAT1 + 1e-9, 0.02)
GL, GA = np.meshgrid(glon, glat)
gx = (GL-LON0)*KM*COS; gy = (GA-LAT0)*KM
B = np.full(GL.shape, np.nan)
for j in range(GL.shape[0]):
    idx = tree.query_ball_point(np.column_stack([gx[j], gy[j]]), R_KM)
    for i, members in enumerate(idx):
        if len(members) >= N_MIN:
            m = mag[members]
            B[j, i] = np.log10(np.e) / (m.mean() - (MC - dM/2.0))   # Aki-Utsu MLE

def nan_smooth(A, passes=2):                  # light NaN-aware 3x3 smoothing
    A = A.copy()
    for _ in range(passes):
        out = A.copy()
        for j in range(A.shape[0]):
            for i in range(A.shape[1]):
                if not np.isfinite(A[j, i]):
                    continue
                sl = A[max(0, j-1):j+2, max(0, i-1):i+2]
                out[j, i] = np.nanmean(sl)
        A = out
    return A
B = nan_smooth(B, passes=2)
Bm = np.ma.masked_invalid(B)
print(f"valid nodes={np.isfinite(B).sum()}/{B.size}  b range={np.nanmin(B):.2f}-{np.nanmax(B):.2f}")

# --- coastline -----------------------------------------------------------
def clip_lines(path, pad=0.25):
    sf = shapefile.Reader(path); out = []
    for sh in sf.shapes():
        pts = sh.points; parts = list(sh.parts) + [len(pts)]
        for i in range(len(parts)-1):
            s = np.array(pts[parts[i]:parts[i+1]])
            if (s[:,0].max()<LON0-pad or s[:,0].min()>LON1+pad or
                    s[:,1].max()<LAT0-pad or s[:,1].min()>LAT1+pad):
                continue
            out.append(s)
    return out
coast = clip_lines(NE_COAST)

# --- plot ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 4.5))
with rasterio.open(TIFF) as ds:
    win = from_bounds(LON0, LAT0, LON1, LAT1, ds.transform).round_offsets().round_lengths()
    arr = ds.read(window=win); bl, bb, br, bt = win_bounds(win, ds.transform)
ax.imshow(np.transpose(arr, (1,2,0)), extent=[bl, br, bb, bt], origin="upper",
          zorder=0, interpolation="bilinear")
# mute the relief a little so the b colours read
ax.add_patch(plt.Rectangle((LON0, LAT0), LON1-LON0, LAT1-LAT0, fc="white", alpha=0.28, lw=0, zorder=0.5))

levels = np.linspace(0.60, 1.20, 13)
cf = ax.contourf(GL, GA, Bm, levels=levels, cmap="RdYlBu", alpha=0.80,
                 extend="both", zorder=1)
ax.contour(GL, GA, Bm, levels=[0.8], colors="#7a0a0a", linewidths=0.8,
           linestyles="--", zorder=2.2)

for s in coast:
    ax.plot(s[:,0], s[:,1], color="#10222e", lw=0.8, zorder=2.4)
for poly, lw in zip(strands, [2.0, 1.5, 1.5]):
    ax.plot(poly[:,0], poly[:,1], color="#111", lw=lw, zorder=3, solid_capstyle="round")

# locked-gap annotation
ax.annotate("locked Marmara gap\n(low $b$)", xy=(28.5, 40.83), xytext=(28.62, 41.20),
            fontsize=8.8, color="#5a0a0a", ha="center", va="top", fontweight="bold",
            path_effects=HALO, zorder=6,
            arrowprops=dict(arrowstyle="->", color="#5a0a0a", lw=1.1))
ax.annotate("1999 \u0130zmit\n(high $b$)", xy=(30.05, 40.78), xytext=(30.05, 41.16),
            fontsize=8.0, color="#0a2a5a", ha="center", va="top",
            path_effects=HALO, zorder=6,
            arrowprops=dict(arrowstyle="->", color="#0a2a5a", lw=1.0))

ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1); ax.set_aspect(1.0/COS)
ax.grid(True, color="white", lw=0.5, ls=(0,(1,3)), alpha=0.5, zorder=1.5)

def dm(v, ew):
    d = int(np.floor(v)); m = int(round((v-d)*60))
    if m == 60: d += 1; m = 0
    return f"{d}\u00b0{m:02d}\u2032{ew}"
ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.25)); ax.yaxis.set_minor_locator(MultipleLocator(0.125))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: dm(v,"E")))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: dm(v,"N")))
ax.tick_params(labelsize=8.5)

# scale bar (top-left) + north arrow (top-right)
barkm = 50.0; bardeg = barkm/(KM*COS); x0, y0 = LON0+0.12, LAT1-0.085
ax.plot([x0, x0+bardeg], [y0, y0], "-", color="white", lw=3.0, zorder=8, solid_capstyle="butt")
ax.text(x0+bardeg/2, y0-0.03, f"{barkm:g} km", ha="center", va="top", color="white",
        fontsize=8.5, zorder=8, path_effects=[pe.withStroke(linewidth=2, foreground="#333")])
nx, ny = LON1-0.13, LAT1-0.32
ax.add_patch(FancyArrow(nx, ny, 0, 0.155, width=0.0, head_width=0.05, head_length=0.05,
                        length_includes_head=True, color="white", zorder=8))
ax.text(nx, ny+0.195, "N", ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="white", zorder=8, path_effects=[pe.withStroke(linewidth=2, foreground="#333")])

# colorbar
div = make_axes_locatable(ax); cax = div.append_axes("right", size="3%", pad=0.12)
cb = fig.colorbar(cf, cax=cax, ticks=np.arange(0.6, 1.21, 0.1))
cb.set_label(r"Gutenberg--Richter $b$-value", fontsize=10)
cb.ax.tick_params(labelsize=8.5)

fig.tight_layout(pad=0.4)
fig.savefig("figures/fig_bvalue_map.pdf", bbox_inches="tight")
fig.savefig("figures/fig_bvalue_map.png", dpi=230, bbox_inches="tight")
print("saved")
