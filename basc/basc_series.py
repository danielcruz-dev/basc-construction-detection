"""
Per-site fraction time series from the Sentinel-2 stacks.

Endmembers are extracted ONCE across all sites (pooled), not per site. Per-site
endmembers would put every site on its own basis and make fractions
incomparable between sites -- the same pooling hazard flagged in
RECOMMENDATION_2026-09-04 s.5, and fatal for a queue that ranks globally. The
paper likewise uses one endmember set for all five cities.

Each period is unmixed and reduced to parcel-level numbers: mean fraction per
endmember, and the AREA whose fraction exceeds tau -- the aggregate-safe stand-in
for the paper's connected-component filter.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from basc_lsma import load_stack, site_pixels, extract_endmembers, unmix, EM_NAMES

ROOT = "results/basc/l2a"
OUT  = "results/basc/series"


def set_paths(root, out):
    global ROOT, OUT
    ROOT, OUT = root, out


def site_dirs():
    return sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))


def pooled_endmembers(dirs, per_site=60000, trim=0.01, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for d in dirs:
        X = site_pixels(d)
        if len(X) == 0:
            continue
        if len(X) > per_site:
            X = X[rng.choice(len(X), per_site, replace=False)]
        pool.append(X)
        print(f"  {os.path.basename(d)[:44]:<44} {len(X):>7} px")
    X = np.vstack(pool)
    M, diag = extract_endmembers(X, seed=seed, trim=trim)
    return M, diag


def aoi_pixels(g):
    """Number of RASTERISED pixels inside the AOI -- the cloud denominator.

    valid_frac is counted against the bounding box, so an irregular AOI can
    never reach 1.0 (a site-plan footprint union tops out near 0.50, a Regrid
    parcel near 0.72, a ground square at 1.00) and one cloud threshold across
    those confounds AOI SHAPE with cloud.

    The denominator is the rasterised PIXEL COUNT, not polygon-area divided by
    bbox-area. The two disagree wherever the AOI edge cuts pixels or the AOI is
    small relative to the grid, and the pixel count is what the imagery
    actually delivers -- an area ratio can imply a fraction of a pixel is
    observable when no such pixel exists.
    """
    try:
        from geo_utils import rasterize, as_geom
        geom = as_geom(g["geometry"])
        m = rasterize(geom, g["bbox"], g["width"], g["height"])
        n = int(m.sum())
        return (n if n > 0 else g["width"] * g["height"]), m
    except Exception:
        return g["width"] * g["height"], None


def run_site(d, M, tau=0.5):
    g = json.load(open(os.path.join(d, "grid.json")))
    px_area = g["res_m_x"] * g["res_m_y"]
    n_aoi_px, aoi_mask = aoi_pixels(g)
    rows = []
    for p in g["periods"]:
        fn = os.path.join(d, f"{p}.tif")
        if not os.path.exists(fn):
            rows.append({"period": p, "n_valid": 0}); continue
        try:
            r, m = load_stack(fn)
        except Exception:
            rows.append({"period": p, "n_valid": 0}); continue
        ok = m & (r > 0).all(-1) & (r < 1.2).all(-1)
        n = int(ok.sum())
        if n < 20:
            rows.append({"period": p, "n_valid": n}); continue
        F, e = unmix(r[ok], M)
        row = {"period": p, "n_valid": n,
               "valid_frac": float(n / (g["width"] * g["height"])),
               "n_aoi_px": int(n_aoi_px),
               # cloud-free share OF THE AOI, comparable across AOI shapes,
               # denominated in rasterised pixels
               "clear_frac": float(min(n / n_aoi_px, 1.0)),
               "rmse": float(np.mean(e))}
        for i, nm in enumerate(EM_NAMES):
            row[nm] = float(F[:, i].mean())
            row[f"{nm}_area_m2"] = float((F[:, i] >= tau).sum() * px_area)
        rows.append(row)
    # start_is_pseudo / class_label are carried through deliberately. A negative's
    # start_period is an ANCHOR drawn from the positives, not an observation; if
    # that distinction is lost here, a downstream reader sees 20 extra "verified
    # starts" and every precision number computed from them is wrong.
    site = {k: g[k] for k in ("uid","name","state","area_ha",
                              "n_buildings","start_period")}
    for k in ("start_is_pseudo", "class_label"):
        if k in g:
            site[k] = g[k]
    return {"site": site,
            "aoi_provenance": g.get("aoi_provenance"),
            "n_aoi_px": int(n_aoi_px),
            "tau": tau, "px_area_m2": px_area, "series": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--trim", type=float, default=0.01)
    ap.add_argument("--endmembers", default="",
                    help="path to an _endmembers.json to REUSE instead of extracting. "
                         "Fractions are only comparable across AOI definitions if every "
                         "run is projected onto the same basis; and a single small AOI "
                         "does not span enough spectral range to place a simplex at all "
                         "(the site-plan run put 'high albedo' at brightness 0.18, "
                         "darker than its own soil).")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    set_paths(a.root, a.out)
    os.makedirs(OUT, exist_ok=True)
    dirs = [d for d in site_dirs() if glob.glob(os.path.join(d, "*.tif"))]
    if a.endmembers:
        src = json.load(open(a.endmembers))["endmembers"]
        M = np.vstack([src[n] for n in EM_NAMES])
        diag = {"reused_from": a.endmembers}
        print(f"reusing endmembers from {a.endmembers}")
    else:
        print(f"pooling endmembers across {len(dirs)} sites")
        M, diag = pooled_endmembers(dirs, trim=a.trim)
        print(f"  {diag['n_pixels']} px -> {diag['n_after_trim']} after {a.trim:.1%} trim")
    print(f"  {'endmember':<14}" + "".join(f"{b:>8}" for b in ["B02","B03","B04","B08","B11","B12"]) + f"{'NDVI':>8}{'bright':>8}")
    for i, n in enumerate(EM_NAMES):
        e = M[i]
        print(f"  {n:<14}" + "".join(f"{v:8.3f}" for v in e)
              + f"{(e[3]-e[2])/(e[3]+e[2]):8.3f}{e.mean():8.3f}")
    json.dump({"endmembers": {n: M[i].tolist() for i, n in enumerate(EM_NAMES)},
               "diag": diag}, open(os.path.join(OUT, "_endmembers.json"), "w"), indent=1)
    for d in dirs:
        print(f"{os.path.basename(d)} ...", flush=True)
        res = run_site(d, M, tau=a.tau)
        json.dump(res, open(os.path.join(OUT, os.path.basename(d) + ".json"), "w"), indent=1)
        u = sum(1 for r in res["series"] if r.get("n_valid", 0) >= 20)
        print(f"  {u} usable periods")


if __name__ == "__main__":
    main()
