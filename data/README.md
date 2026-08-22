# Base-map data (not redistributed)

Two of the figure scripts (`script_epicentres.py`, `script_bvalue_map.py`) draw a
geographic base and therefore need two data files placed in this `data/` folder.
They are **not** included in the repository (licensing / file size); download them
and drop them here with exactly these names:

| File                     | What it is                                   | Where to get it |
|--------------------------|----------------------------------------------|-----------------|
| `exportImage.tiff`       | Topo–bathymetric relief raster, EPSG:4326, covering ≈25–31°E / 39–42°N | Any WGS84 GeoTIFF relief export for the window (e.g. GEBCO, ETOPO1, or a basemap export). |
| `ne_10m_coastline.shp`   | Natural Earth 1:10m physical coastline (plus the sibling `.shx`, `.dbf`, `.prj`) | https://www.naturalearthdata.com/downloads/10m-physical-vectors/ (public domain) |

The `.tif/.tiff` rasters are git-ignored by default (see `.gitignore`) so you do not
accidentally commit a large binary. The four non-map figures (Figs 1, 3, 4) and the
correlation-integral figure run without any of these files — the earthquake catalogue
is simulated from fixed random seeds so every figure is reproducible as-is.
