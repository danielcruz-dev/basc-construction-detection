"""Select negatives area-matched to the sp16 positives' PARCEL areas.

Comparison A is parcel-vs-parcel, so the match must be on parcel area -- the
geometry both classes will actually be fetched with -- not on the positives'
site-plan AOI, which the negatives can never have.
"""
import json, os, sys, glob, math, random, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
from sites import load_sites
from replay import load_truth, ordn, unordn
try:
    # acd-polygon-lab, when it is on PYTHONPATH
    from construction_start_backtest import ALL_STAGES
except ImportError:
    # The negatives sit in pre-start stages that load_sites() filters out by
    # default, so the pool must be loaded with every stage visible. Mirrored
    # here so this repo does not hard-depend on the lab checkout.
    ALL_STAGES = {"announcement", "construction", "active", "delayed",
                  "land bank", "not approved withdrawn", "cancelled"}

SEED = 20260908
N_NEG = 20
SCR = os.environ.get("BASC_PICK_OUT", "results")   # where the pick lists land

# --- positives: the 16 already fetched at site-plan AOI ---
pos = {}
for gf in sorted(glob.glob("results/sp16/*/grid.json")):
    j = json.load(open(gf))
    pos[j["uid"]] = j["start_period"]
print(f"sp16 positives: {len(pos)}")

truth = load_truth()
sites = {s.unit_uid: s for s in load_sites(stages=ALL_STAGES)}
print(f"inventory loaded: {len(sites)}")

def parcel_ha(uid):
    s = sites.get(uid)
    if s is None or s.geometry is None or s.geometry.is_empty: return None
    return (getattr(s, "area_m2", 0) or 0) / 1e4

pos_area = {u: parcel_ha(u) for u in pos}
missing = [u for u, v in pos_area.items() if not v]
if missing: print("positives with no parcel area:", missing)
PA = sorted(v for v in pos_area.values() if v)
print(f"positive PARCEL ha: min {PA[0]:.1f} med {st.median(PA):.1f} max {PA[-1]:.1f}")

# --- negative pool ---
NEG_POOL = os.environ.get(
    "BASC_NEG_POOL",
    "/home/azul/Aterio/acd-polygon-lab/results/negative_parcels.json")
negs = json.load(open(NEG_POOL))["uids"]
pool = []
for u in negs:
    if u in truth:  # never a negative
        continue
    a = parcel_ha(u)
    if a and a > 0:
        pool.append((u, a))
print(f"negative pool with parcel area: {len(pool)}")

# --- 1:1 nearest-neighbour on log area, then KS-greedy extras ---
# The first pass pairs each positive with its closest-area negative, which is
# what makes comparison A a matched design. Naively cycling the same ascending
# order for the extras drew all four off the small end and pulled the negative
# median to 0.63x the positive one; the extras are instead chosen to minimise
# the two-sample KS distance on log area, so they close the gap rather than widen it.
rng = random.Random(SEED)
order = sorted(pos_area.items(), key=lambda kv: kv[1] or 0)
used, chosen = set(), []
for puid, pa in order:
    cands = [(abs(math.log(a) - math.log(pa)), u, a) for u, a in pool if u not in used]
    if not cands: break
    d, u, a = min(cands)
    used.add(u); chosen.append((u, a, puid, pa))

def ks(xs, ys):
    xs, ys = sorted(xs), sorted(ys)
    allv = sorted(set(xs) | set(ys))
    F = lambda v, z: sum(1 for t in z if t <= v) / len(z)
    return max(abs(F(v, xs) - F(v, ys)) for v in allv)

logpos = [math.log(v) for v in PA]
while len(chosen) < N_NEG:
    cur = [math.log(a) for _, a, _, _ in chosen]
    best = None
    for u, a in pool:
        if u in used: continue
        k = ks(cur + [math.log(a)], logpos)
        if best is None or k < best[0]: best = (k, u, a)
    if best is None: break
    _, u, a = best
    used.add(u); chosen.append((u, a, None, None))

print(f"\nKS(log area) neg vs pos = {ks([math.log(a) for _,a,_,_ in chosen], logpos):.3f}")

# --- pseudo-starts: each matched pair SHARES a start period ---
# A negative has no construction, so its start period is only an anchor for the
# season-matched window. Giving the pair the same anchor makes the design paired
# on season as well as area, which is what neutralises the February visibility
# bias in log section 5 -- an independent resample left 2025-06-H1 with four
# negatives against one positive, reintroducing exactly that imbalance.
starts_pool = sorted(pos.values())
pseudo = {}
for u, a, puid, pa in chosen:
    pseudo[u] = pos[puid] if puid else rng.choice(starts_pool)

NA = sorted(a for _, a, _, _ in chosen)
print(f"\nchosen negatives: {len(chosen)}")
print(f"negative PARCEL ha: min {NA[0]:.1f} med {st.median(NA):.1f} max {NA[-1]:.1f}")
print(f"median ratio neg/pos = {st.median(NA)/st.median(PA):.2f}x")

print(f"\n{'negative':<10}{'ha':>9}   matched to positive{'':<14}{'ha':>9}   pseudo-start")
for u, a, puid, pa in chosen:
    pm = sites[puid].unit_name[:26] if puid else "(KS-greedy extra)"
    pas = f"{pa:9.1f}" if pa else "        —"
    print(f"{u[:8]:<10}{a:9.1f}   {pm:<30}{pas}   {pseudo[u]}")

from collections import Counter
print("\npseudo-start distribution vs positives:")
cp, cn = Counter(pos.values()), Counter(pseudo.values())
for p in sorted(set(cp) | set(cn)):
    print(f"  {p}   pos {cp.get(p,0)}   neg {cn.get(p,0)}")

json.dump([u for u, _, _, _ in chosen], open(f"{SCR}/picks_neg20.json", "w"), indent=1)
json.dump(pseudo, open(f"{SCR}/starts_neg20.json", "w"), indent=1)
json.dump(sorted(pos), open(f"{SCR}/picks_pos16.json", "w"), indent=1)
pairs = {u: puid for u, a, puid, pa in chosen if puid}
json.dump(pairs, open(f"{SCR}/pairs_neg20.json", "w"), indent=1)
meta = {"seed": SEED, "n_negatives": len(chosen), "n_pairs": len(pairs),
        "ks_log_area": ks([math.log(a) for _, a, _, _ in chosen], logpos),
        "pos_parcel_ha_median": st.median(PA),
        "neg_parcel_ha_median": st.median([a for _, a, _, _ in chosen]),
        "pseudo_starts": pseudo,
        "note": "negatives: stage in {Announcement,Land Bank,Delayed}, no verified "
                "start, has polygon. 16 matched 1:1 to an sp16 positive on log parcel "
                "area and sharing its start period; 4 KS-greedy extras close the area "
                "distribution and are unpaired."}
json.dump(meta, open(f"{SCR}/negatives_design.json", "w"), indent=1)
print(f"wrote pairs_neg20.json ({len(pairs)} pairs) and negatives_design.json")
print(f"\nwrote picks_neg20.json ({len(chosen)}), starts_neg20.json, picks_pos16.json ({len(pos)})")
