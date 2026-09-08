"""Flatten the series into one labelling / training table.

One row per campus per fortnight: the chip you would look at, the features
already computed from that exact raster, and the label as it currently stands.

The point is that a human labelling a stage and a model training on the same row
see the SAME evidence. Nothing here is re-derived from a different mask, a
different AOI or a different endmember basis -- the recurring failure in this
project (2026-09-07 section 4) is numbers that were compared across bases without
anyone noticing.

`offset` is fortnights from the campus's start anchor, so a stage labeller can
sort by it, and `start_is_pseudo` says whether that anchor is an observation or a
drawn one. Rows from a negative campus are labelled cls=negative and MUST NOT be
read as "stage 0 construction".

  python export_labeling_set.py --series results/series_pos16par:positive \
                                --series results/series_neg20:negative \
                                --chips  results/chips_pos16par \
                                --out    results/labeling_set.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

from replay import ordn

FRACTIONS = ["vegetation", "soil", "high_albedo", "low_albedo"]
INDICES = ["ndvi", "ndbi", "swir"]
AREAS = ["soil_area_m2", "high_albedo_area_m2"]

COLS = (["uid", "slug", "name", "state", "area_ha", "n_buildings",
         "cls", "start_period", "start_is_pseudo", "review_type",
         "period", "offset", "n_valid", "clear_frac", "rmse"]
        + FRACTIONS + INDICES + AREAS + ["chip"])


def rows_for(series_dir, cls, chipdir, types):
    out = []
    for fn in sorted(glob.glob(f"{series_dir}/*.json")):
        if os.path.basename(fn).startswith("_"):
            continue
        j = json.load(open(fn))
        s = j["site"]
        slug = os.path.basename(fn)[:-5]
        o0 = ordn(s["start_period"])
        for r in j["series"]:
            if r.get("n_valid", 0) < 20:
                continue
            chip = os.path.join(chipdir, slug, f"{r['period']}.png") if chipdir else ""
            row = {
                "uid": s.get("uid", ""), "slug": slug, "name": s["name"],
                "state": s.get("state", ""), "area_ha": round(s.get("area_ha", 0), 2),
                "n_buildings": s.get("n_buildings", ""),
                "cls": cls,
                "start_period": s["start_period"],
                "start_is_pseudo": bool(s.get("start_is_pseudo")),
                "review_type": types.get(s.get("uid", ""), ""),
                "period": r["period"], "offset": ordn(r["period"]) - o0,
                "n_valid": r["n_valid"],
                "clear_frac": round(r.get("clear_frac", r.get("valid_frac", 0)), 4),
                "rmse": round(r.get("rmse", 0), 5),
                "chip": chip if chip and os.path.exists(chip) else "",
            }
            for k in FRACTIONS + INDICES + AREAS:
                v = r.get(k)
                row[k] = round(v, 5) if isinstance(v, (int, float)) else ""
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="seriesdir:cls  (cls is positive or negative)")
    ap.add_argument("--chips", action="append", default=[],
                    help="chip dir, positionally matched to --series")
    ap.add_argument("--corrections", default="results/acd_ui_corrections.json")
    ap.add_argument("--out", default="results/labeling_set.csv")
    a = ap.parse_args()

    types = {}
    if os.path.exists(a.corrections):
        types = json.load(open(a.corrections)).get("types", {})

    allrows = []
    for i, spec in enumerate(a.series):
        sdir, cls = spec.rsplit(":", 1)
        chipdir = a.chips[i] if i < len(a.chips) else ""
        rs = rows_for(sdir, cls, chipdir, types)
        allrows += rs
        camp = len({r["uid"] for r in rs})
        withchip = sum(1 for r in rs if r["chip"])
        print(f"  {os.path.basename(sdir):<22} {cls:<9} {camp:>3} campuses  "
              f"{len(rs):>5} rows  {withchip:>5} with a chip")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(allrows)
    print(f"\nwritten {a.out}  {len(allrows)} rows, {len(COLS)} columns")
    lab = sum(1 for r in allrows if r["review_type"])
    print(f"rows carrying a reviewer construction type: {lab}")


if __name__ == "__main__":
    main()
