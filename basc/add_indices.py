"""NDVI/NDBI/SWIR from the same rasters and the same mask as the fractions."""
import argparse, glob, json, os
import numpy as np
from basc_lsma import load_stack
ap=argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--series")
a=ap.parse_args()
for fn in sorted(glob.glob(f"{a.series}/*.json")):
    if fn.endswith("_endmembers.json"): continue
    d=json.load(open(fn)); slug=os.path.basename(fn)[:-5]; n=0
    for row in d["series"]:
        f=os.path.join(a.root, slug, f"{row['period']}.tif")
        if not os.path.exists(f): continue
        try: r,m=load_stack(f)
        except Exception: continue
        ok=m&(r>0).all(-1)&(r<1.2).all(-1)
        if ok.sum()<20: continue
        v=r[ok]; red,nir,sw=v[:,2],v[:,3],v[:,4]
        row["ndvi"]=float(np.mean((nir-red)/(nir+red+1e-6)))
        row["ndbi"]=float(np.mean((sw-nir)/(sw+nir+1e-6)))
        row["swir"]=float(np.mean(sw)); n+=1
    json.dump(d,open(fn,"w"),indent=1)
    print(f"  {slug[:44]:<44} {n}")
