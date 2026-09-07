"""
Truecolor thumbnails rendered FROM the analysis stacks.

Not a fresh API fetch: these are the same arrays the fractions and indices are
computed from, so what you see is literally what was measured.

The stretch is computed ONCE per site over every period and then applied
unchanged to all of them. Per-image autoscaling would make brightness changes an
artefact of the renderer -- the exact thing this strip exists to let you check.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from PIL import Image
from basc_lsma import load_stack

ROOT, OUT = "results/basc/l2a", "results/basc/chips"


def set_paths(root, out):
    global ROOT, OUT
    ROOT, OUT = root, out
THUMB_H = 108

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); set_paths(a.root, a.out)
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(d): continue
        gp = os.path.join(d, "grid.json")
        if not os.path.exists(gp):
            continue          # site skipped upstream (e.g. no site plan for it)
        g = json.load(open(gp))
        slug = os.path.basename(d)
        od = os.path.join(OUT, slug); os.makedirs(od, exist_ok=True)

        # one stretch for the whole site, from a sample of valid pixels
        samp = []
        for fn in sorted(glob.glob(os.path.join(d, "*.tif"))):
            try: r, m = load_stack(fn)
            except Exception: continue
            v = r[m & (r > 0).all(-1) & (r < 1.2).all(-1)]
            if len(v): samp.append(v[::max(1, len(v)//2000), :3])
        if not samp:
            print(f"{slug}: no valid pixels"); continue
        S = np.vstack(samp)
        lo = np.quantile(S, 0.02, axis=0); hi = np.quantile(S, 0.98, axis=0)

        n = 0
        for p in g["periods"]:
            fn = os.path.join(d, f"{p}.tif")
            out = os.path.join(od, f"{p}.png")
            if not os.path.exists(fn): continue
            try: r, m = load_stack(fn)
            except Exception: continue
            ok = m & (r > 0).all(-1) & (r < 1.2).all(-1)
            if ok.sum() < 20: continue
            bgr = r[..., :3]                      # B02,B03,B04 -> B,G,R
            rgb = np.stack([bgr[..., 2], bgr[..., 1], bgr[..., 0]], -1)
            l3, h3 = lo[::-1], hi[::-1]
            img = np.clip((rgb - l3) / np.maximum(h3 - l3, 1e-6), 0, 1)
            img = np.power(img, 1/1.25)           # mild gamma, keeps shadows readable
            a = (img * 255).astype(np.uint8)
            a[~ok] = 26                           # masked -> near-surface grey
            im = Image.fromarray(a, "RGB")
            w, h = im.size
            im = im.resize((max(1, round(w * THUMB_H / h)), THUMB_H), Image.NEAREST)
            im.save(out, optimize=True)
            n += 1
        print(f"{slug[:46]:<46} {n} chips  ({im.size[0]}x{THUMB_H})")

if __name__ == "__main__":
    main()
