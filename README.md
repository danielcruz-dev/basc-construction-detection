# THE AMAZIN ALOGRITHM

Placed here 2026-09-07. These are **copies**. The originals are untouched and still in
place — see "Do not treat this as the deployment copy" below.

## What this is

`aterio_data_center_power_capacity_estimator.pkl` — the most structurally complex
trained model built in-house that exists as a file on this machine.

| property | value |
|---|---|
| class | `sklearn.ensemble.RandomForestRegressor` |
| trees | 100 |
| total nodes | **71,806** |
| mean tree depth | 27.8 |
| max tree depth | 37 |
| `n_features_in_` | **118** |
| file size | 5,199,974 bytes |
| md5 | `42baa91076e07b427a6ff6b5a702b5c7` |

`aterio_data_center_power_capacity_estimator_encoder.pkl` (2,587 B, md5
`65b264243166713a855902827b8ea25e`) is copied alongside it **because the model is
unusable without it**. The regressor takes 118 columns; those 118 come out of the
encoder. Handed the raw fields instead, it will not error — it will silently predict
against a mis-aligned feature vector.

## Original location

```
~/Aterio/aterio-gcp-data-operations/gcp-cloud-run/services/data-centers/
  data-process-data-centers-api/app/models/
```

**Do not treat this as the deployment copy.** That path is a live Cloud Run service's
model directory, which is why these were copied rather than moved. There is no write
access to `gcp-data-operations` from either gh account, so the deployed artefact can
only change by the user pushing it.

## Loading it — version caveat

The pickle was written by **scikit-learn 1.5.1** (the estimator) / **1.5.2** (the
pipeline objects). The `geo` env here is on **1.9.0**, which loads it but raises
`InconsistentVersionWarning`. That warning is not decorative on tree ensembles — pin
1.5.x if the predictions matter:

```python
import pickle
m = pickle.load(open("aterio_data_center_power_capacity_estimator.pkl", "rb"))
```

## How "most complex" was decided, and what it excludes

Ranked by learned structure (nodes / parameters), not by file size, across the whole
machine including Trash and caches.

**Not chosen — bigger, but not yours.** All third-party downloads:

| | size | note |
|---|---|---|
| LAION `CLIP-ViT-B-32-laion2B-s34B-b79K` | 578 MB cache | ~151M params — by raw parameter count this is the most complex model on the machine, full stop |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 78 MB | torch hub cache |
| `resnet18` | 47 MB | torch hub cache |
| Chrome / Brave `*.tflite` | ~1–37 MB each | browser built-ins (OCR, optimisation guide) |

**Not chosen — more complex in configuration, but not a file.** `acd-polygon-lab/
train_model.py` fits a `RandomForestClassifier(n_estimators=400, min_samples_leaf=2)`
and an `MLPClassifier(hidden_layer_sizes=(64, 32))`. That forest is 4x the trees of the
one here, but the script **never persists a fitted model** — it writes only JSON metrics
(`results/model_comparison*.json`). There is no artefact to copy. If the intended
"amazing algorithm" is the ACD work rather than the capacity estimator, the thing to
capture is the *code plus a re-fit*, not a file that does not exist yet.

**Also not chosen — the sibling estimator.**
`aterio_data_center_power_capacity_hp_estimator.pkl`: `Pipeline(ColumnTransformer →
RandomForestRegressor)`, 100 trees but only 20,400 nodes, max depth 23, 6 raw input
columns. Strictly smaller. It is left in place; say the word if you want it here too.
Note it needs a `sklearn.compose._column_transformer._RemainderColsList` shim to unpickle
on 1.9.0.

---

# BASC replication — construction-start detection (added 2026-09-07)

Replication of Chen et al., *"Broad-area-search of new construction using time series
analysis of Landsat and Sentinel-2 data"*, Science of Remote Sensing, May 2024
(<https://www.sciencedirect.com/science/article/pii/S2666017224000221>), applied to
Aterio's data-centre sites on Sentinel-2.

## What runs

| file | does |
|---|---|
| `basc/basc_fetch.py` | per-pixel Sentinel-2 stacks (6 bands + CLP + validity) from the CDSE Process API. `--aoi-m 400` uses a ground square on the site point; omit it for the Regrid parcel |
| `basc/basc_lsma.py` | fully constrained LSMA. Endmembers are extracted from the imagery (N-FINDR on PCs, winsorised, vertex = median of 200 nearest) — see caveat below |
| `basc/basc_series.py` | unmixes every period to parcel/AOI-level fractions + area above threshold |
| `basc/basc_chips.py` | truecolor thumbnails from the *same* arrays, one fixed stretch per site |
| `basc/results/fraction_timeseries.html` | the viewer: fractions vs indices, imagery strip, parcel/AOI toggle |

Open the HTML over http (`python -m http.server` in `basc/results/`), not `file://` —
the chips are loaded by relative path.

## Endmembers: why they are not Wu & Murray's numbers

The paper says it uses "the same endmember values introduced in Wu and Murray (2003)".
Those values are **not published as numbers** — Wu & Murray print them only as a plot
(their Fig. 5), and they were image-derived from feature-space extremes of a single
July-1999 Columbus, Ohio ETM+ scene. So this replicates their *method* rather than
transplanting spectra from a 1999 Ohio scene onto 2025 Sentinel-2 over Iowa and
Virginia. Endmembers are pooled across sites so fractions stay comparable between them.

## Deviations from the paper, deliberate

- **Sentinel-2 L2A**, not the L1C TOA the paper used for S2 (they used TOA only because
  GEE lacks L2A before 2019). Fractions will not be numerically identical to theirs.
- **No CCDC.** CCDC needs ~2 years of history to initialise and a fitted *post-break*
  segment for `Cdif` / `Paft`; five of the paper's seven rules therefore cannot run in
  near-real-time. The paper's own success criterion is a detection within **two years**
  of the true date. Aterio needs a fortnight.
- **Cloud**: SCL classes 4/5/6/7 **and** s2cloudless `CLP` <= 65%, as the paper does.
  Measured effect: drops 0.6-9.6% of pixels, cuts spurious high-albedo by 23-33% on
  cloudy periods, moves clean periods by <0.001.
- **Tile filter left loose** (`maxCloudCoverage=100`). It is measured over the whole
  ~110 km granule, not the AOI; tightening it only creates gaps. Tested: relaxing it
  40 -> 100 recovered only 5 of 31 missing periods on one site.

## External dependencies not in this repo

`sites.py` reads `~/Aterio/parcels-run/` (parcel geometries + inventory) and `replay.py`
reads `acd-polygon-lab/results/acd_ui_corrections.json` (the hand-verified start dates)
and `stats_cache.json`. Those are not copied here. CDSE credentials come from
`~/.cdse.env`.

## Status

Five sites, exploratory. **No controls yet** — without the non-starting parcels none of
this supports a claim about detection, only about mechanism.
