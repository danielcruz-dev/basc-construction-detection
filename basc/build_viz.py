"""
Build the fraction/index viewer from any number of AOI variants.

Each variant is one AOI definition run through the same pipeline, so the page
can switch between them and the effect of the footprint choice is visible
rather than argued about.

  python build_viz.py \
      --variant aoi400:results/series_aoi400:results/chips_aoi400:"400 m box" \
      --variant parcel:results/series:results/chips:"Regrid parcel"
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image

from replay import ordn

NOTES = {
    "siteplan": "Union of the campus's PLANNED building footprints from the "
                "georeferenced site plan, buffered 100 m. The sheet predates the "
                "verified start, so this is information a detector could have had "
                "before construction — unlike a footprint traced from later imagery.",
    "aoi400":   "400 m ground square on the site point — uniform 40×40 px, 8,000 m² "
                "is 5% of it, no dependence on Regrid. Tight, but it clips large "
                "campuses: only 26% of Digital's change and none of QTS's fell inside.",
    "aoi1000":  "1,000 m ground square on the site point — 100 ha. Captures far more "
                "of a spread-out campus, at the cost of sensitivity: 8,000 m² is now "
                "only 0.8% of the AOI.",
    "parcel":   "The Regrid parcel polygon — 11 to 320 ha across these five, irregular, "
                "SCL-only cloud masking, and 189 of 1,209 sites have no usable parcel.",
}


def build(serdir: str, chipdir: str) -> list[dict]:
    out = []
    for fn in sorted(glob.glob(f"{serdir}/*.json")):
        if fn.endswith("_endmembers.json"):
            continue
        d = json.load(open(fn))
        s = d["site"]
        slug = os.path.basename(fn)[:-5]
        per = [r["period"] for r in d["series"]]
        o0 = ordn(per[0])
        rows = []
        for r in d["series"]:
            k = ordn(r["period"]) - o0
            if r.get("n_valid", 0) < 20:
                rows.append({"x": k, "p": r["period"]})
                continue
            rows.append({
                "x": k, "p": r["period"], "v": round(r["valid_frac"], 3),
                "veg": round(r["vegetation"], 4), "soil": round(r["soil"], 4),
                "high": round(r["high_albedo"], 4), "low": round(r["low_albedo"], 4),
                "sha": round(r["soil_area_m2"] / 1e4, 2),
                "hha": round(r["high_albedo_area_m2"] / 1e4, 2),
                "ndvi": round(r["ndvi"], 4) if "ndvi" in r else None,
                "ndbi": round(r["ndbi"], 4) if "ndbi" in r else None,
                "swir": round(r["swir"], 4) if "swir" in r else None,
                "rmse": round(r["rmse"], 4),
            })
        chips = sorted(os.path.basename(p)[:-4]
                       for p in glob.glob(f"{chipdir}/{slug}/*.png"))
        site = {"name": s["name"], "state": s["state"], "ha": round(s["area_ha"], 1),
                "bld": s["n_buildings"], "start": s["start_period"],
                "startx": ordn(s["start_period"]) - o0, "periods": per,
                "rows": rows, "slug": slug, "chips": chips}
        if chips:
            # explicit dimensions: a lazy image with no width lets its alt text
            # lay itself out, which shoves the whole filmstrip sideways
            site["cw"], site["ch"] = Image.open(f"{chipdir}/{slug}/{chips[0]}.png").size
        else:
            site["cw"], site["ch"] = 1, 1
        out.append(site)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True,
                    help="key:seriesdir:chipdir:label")
    ap.add_argument("--out", default="results/fraction_timeseries.html")
    ap.add_argument("--template", default="viz_template.html")
    a = ap.parse_args()

    D = {"variants": {}}
    for v in a.variant:
        key, serdir, chipdir, label = v.split(":", 3)
        sites = build(serdir, chipdir)
        emf = os.path.join(serdir, "_endmembers.json")
        D["variants"][key] = {
            "label": label,
            "chipdir": os.path.relpath(chipdir, os.path.dirname(a.out)),
            "note": NOTES.get(key, ""),
            "sites": sites,
            "endmembers": json.load(open(emf))["endmembers"] if os.path.exists(emf) else {},
        }
        print(f"  {key:<10} {len(sites)} sites, "
              f"{sum(len(s['chips']) for s in sites)} chips")

    html = open(a.template).read().replace("__DATA__", json.dumps(D))
    # the template's default variant must actually exist
    first = list(D["variants"])[0]
    html = html.replace('let VAR = "aoi400";', f'let VAR = "{first}";')
    open(a.out, "w").write(html)
    print(f"written {a.out}  ({os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
