"""
Per-pixel Sentinel-2 stacks for a BASC replication test.

Wu & Murray (2003) LSMA needs six bands; acd_core.STACK_EVALSCRIPT omits B12,
so a local evalscript adds it rather than editing the shared one (build_dataset
depends on the existing output shape).

Output per site: results/basc/<slug>/<period>.tif  float32, 7 bands
  B02 B03 B04 B08 B11 B12 valid
plus grid.json with bbox / shape / period list.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, re, threading, time
from acd_core import Cdse, CDSE_PROCESS_URL, MAX_CLOUD_COVERAGE, period_bounds
from replay import load_truth, unordn, ordn, period_of
from sites import load_sites, ground_square

MAXCC = [100]
from shapely.geometry import mapping

# CLP is s2cloudless cloud probability (0-255). SCL alone misses thin cirrus --
# measured: a period SCL called cloud-free carried CLP up to 185, and masking on
# it cut spurious high-albedo by a third on cloudy periods while moving clean
# ones by <0.001. The paper masks on the QA band AND s2cloudless at 65%.
EVAL = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B02","B03","B04","B08","B11","B12","SCL","CLP","dataMask"]}],
    output: {bands: 8, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) {
  var valid = (s.SCL === 4 || s.SCL === 5 || s.SCL === 6 || s.SCL === 7) ? 1 : 0;
  return [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12, s.CLP, s.dataMask * valid];
}"""

OUT = "results/basc"
def slug(s): return re.sub(r"[^a-z0-9]+","-",s.lower()).strip("-")[:60]

def fetch(cdse, bbox, geom, start, end, w, h, collection):
    payload = {
      "input": {
        "bounds": {"bbox": list(bbox), "geometry": geom,
                   "properties": {"crs":"http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
        "data": [{"type": collection,
                  "dataFilter": {"maxCloudCoverage": MAXCC[0],
                                 "mosaickingOrder": "leastCC",
                                 "timeRange": {"from": f"{start}T00:00:00Z",
                                               "to": f"{end}T00:00:00Z"}}}]},
      "output": {"width": w, "height": h,
                 "responses": [{"identifier":"default","format":{"type":"image/tiff"}}]},
      "evalscript": EVAL}
    return cdse.post(CDSE_PROCESS_URL, payload, expect_json=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="/tmp/claude-1000/-home-azul/9d997b7f-bc55-4b65-af21-7ff969074d59/scratchpad/pick5.json")
    ap.add_argument("--from-period", default="2023-01-H1")
    ap.add_argument("--to-period",   default="2026-09-H1")
    ap.add_argument("--collection",  default="sentinel-2-l2a")
    ap.add_argument("--tag",         default="l2a")
    ap.add_argument("--res-m", type=float, default=10.0)
    ap.add_argument("--max-px", type=int, default=1000)
    ap.add_argument("--aoi-m", type=float, default=0.0,
                    help="if >0, use a ground square of this side on the site point "
                         "instead of the Regrid parcel")
    ap.add_argument("--max-cloud", type=int, default=100,
                    help="tile-level metadata filter; loose by default because it is "
                         "measured over the whole ~110km granule, not the AOI")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    MAXCC[0] = a.max_cloud
    picks = json.load(open(a.picks))
    print("loading sites (~2 min) ...", flush=True)
    sites = {s.unit_uid: s for s in load_sites()}
    truth = load_truth()
    periods = [unordn(o) for o in range(ordn(a.from_period), ordn(a.to_period)+1)]
    print(f"{len(periods)} periods {periods[0]}..{periods[-1]} | collection {a.collection}")

    jobs = []
    for uid in picks:
        s = sites[uid]
        d = os.path.join(OUT, a.tag, slug(s.unit_name)); os.makedirs(d, exist_ok=True)
        geom = ground_square(s.lat, s.lon, a.aoi_m) if a.aoi_m > 0 else s.geometry
        x0,y0,x1,y1 = geom.bounds
        import math
        lat = (y0+y1)/2
        wm = (x1-x0)*111320.0*math.cos(math.radians(lat)); hm = (y1-y0)*110540.0
        w = min(a.max_px, max(16, round(wm/a.res_m))); h = min(a.max_px, max(16, round(hm/a.res_m)))
        json.dump({"uid":uid,"name":s.unit_name,"state":s.state_code,
                   "area_ha":s.area_m2/1e4,"n_buildings":s.n_buildings,
                   "start_period":unordn(truth[uid]),"bbox":[x0,y0,x1,y1],
                   "width":w,"height":h,"res_m_x":wm/w,"res_m_y":hm/h,
                   "periods":periods,"collection":a.collection,
                   "bands":["B02","B03","B04","B08","B11","B12","CLP","valid"],
                   "aoi_m":a.aoi_m, "max_cloud":a.max_cloud,
                   "lat":s.lat, "lon":s.lon,
                   "geometry":mapping(geom)}, open(os.path.join(d,"grid.json"),"w"))
        print(f"  {s.unit_name[:44]:<44} {s.area_ha if hasattr(s,'area_ha') else s.area_m2/1e4:7.1f} ha  "
              f"{w}x{h} px  ({wm/w:.1f} m/px)")
        for p in periods:
            fn = os.path.join(d, f"{p}.tif")
            if not os.path.exists(fn): jobs.append((s, geom, p, fn, w, h))
    print(f"{len(jobs)} rasters to fetch")
    if a.dry_run or not jobs: return

    cdse = Cdse(); lock = threading.Lock(); done=[0]; failed=[]
    def work(j):
        s,geom,p,fn,w,h = j
        st,en = period_bounds(p)
        try:
            tif = fetch(cdse, geom.bounds, mapping(geom), st, en, w, h, a.collection)
            tmp = fn+".part"; open(tmp,"wb").write(tif); os.replace(tmp,fn)
        except Exception as e:
            failed.append((s.unit_name,p,str(e)[:100]))
        with lock:
            done[0]+=1
            if done[0]%25==0 or done[0]==len(jobs):
                print(f"  {done[0]}/{len(jobs)} calls={cdse.process_calls} failed={len(failed)}",flush=True)
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(work,jobs))
    print(f"done: {cdse.process_calls} calls, {len(failed)} failed, {(time.time()-t0)/60:.1f} min")
    for f in failed[:15]: print("  FAIL",f)

if __name__=="__main__": main()
