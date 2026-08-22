# Fractal Analysis of Seismicity and Fault Network Geometry — R & Python Workflows

<!-- After Zenodo mints a DOI, replace XXXXXXX with your concept DOI (see step 6 of the setup guide) -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible code accompanying the paper:

> **Lemenkova P., Zülfikar A. C.**, 2026: *Fractal Analysis of Seismicity and Fault Network
> Geometry for Quantifying Earthquake Scale Invariance with R and Python Workflows.*
> Submitted to *Contributions to Geophysics and Geodesy*.

The workflow quantifies the scale invariance of an earthquake catalogue through complementary
measures — the box-counting dimension (*D₀*) and correlation dimension (*D₂*) of the epicentre
distribution, the box-counting dimension of the fault-trace network (*D_f*), and the
maximum-likelihood Gutenberg–Richter *b*-value — together with magnitude-of-completeness
selection and Gardner–Knopoff declustering. It is demonstrated on the Marmara region of the
North Anatolian Fault Zone (NAFZ).

## Repository layout

```
.
├── scripts/      Python scripts that generate the manuscript figures
├── figures/      Rendered figures (PNG)
├── data/         Base-map inputs for the two map figures (see data/README.md)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Figures and the scripts that produce them

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 1 | `scripts/script_fig_01.py`               | Completeness-magnitude selection by *b*-value stability |
| Fig. 2 | `scripts/script_epicentres.py`           | Declustered epicentre distribution (needs base map) |
| Fig. 3 | `scripts/script_correlation_integral.py` | Grassberger–Procaccia correlation integral → *D₂* |
| Fig. 4 | `scripts/script_fmd.py`                  | Frequency–magnitude distribution and *b*-value fit |
| Fig. 5 | `scripts/script_bvalue_map.py`           | Spatial *b*-value map along the Marmara segment (needs base map) |
| Fig. 6 | *(figure only)* `figures/fig_fault_boxcount.png` | Box-counting of the NAFZ fault-trace network → *D_f* |

> **Note:** the plotting script for Fig. 6 is not yet in this repository — only the rendered
> figure is included. Add the script here to make the figure fully reproducible.

The earthquake catalogue is simulated from fixed random seeds inside each script, so the figures
are reproducible without a private data file. Replace the simulation block with your real
KOERI/AFAD/ISC catalogue for the production analysis.

## Requirements

Python 3.10+ and:

```
numpy • scipy • matplotlib • rasterio • pyshp
```

Install into a fresh environment:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

From the repository root (so the relative `figures/` and `data/` paths resolve):

```bash
python scripts/script_fig_01.py
python scripts/script_correlation_integral.py
python scripts/script_fmd.py
python scripts/script_epicentres.py     # requires data/ base map (see data/README.md)
python scripts/script_bvalue_map.py     # requires data/ base map (see data/README.md)
```

Each script writes a `.pdf` and `.png` into `figures/`.

## Data dependencies

`script_epicentres.py` and `script_bvalue_map.py` render a geographic base and need a relief
GeoTIFF (`data/exportImage.tiff`) and the Natural Earth 1:10m coastline
(`data/ne_10m_coastline.shp` + siblings). See [`data/README.md`](data/README.md) for sources.
These are not redistributed here.

## Citation

If you use this code, please cite both the paper (above) and the archived software release
(the Zenodo DOI badge at the top). Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). The Natural Earth coastline is public domain;
the relief raster you supply retains its own license.
