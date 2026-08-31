"""Post-hoc confound analysis on CACHED body-arm embeddings. Read-only."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "/root/SanBox/prototypes/01-melops-ablation")
import melops_data, protocol, emb_cache

root = "/root/data/Melops"
df = melops_data.load_melops(root, bbox="body")
g, q = protocol.one_shot_open_set_split(df, cutoff_fraction=0.5, seed=0)
ge = emb_cache.load("/root/emb_cache", "megadescriptor", "body", emb_cache._ids_list(g["image_id"].tolist()))
qe = emb_cache.load("/root/emb_cache", "megadescriptor", "body", emb_cache._ids_list(q["image_id"].tolist()))
assert ge is not None and qe is not None, "cache miss"

S = qe @ ge.T
gside = g["side"].to_numpy(); qside = q["side"].to_numpy()
mask = qside[:, None] == gside[None, :]
S = np.where(mask, S, -np.inf)
mx = S.max(axis=1); am = S.argmax(axis=1)

meta = pd.read_csv(root + "/Melops_metadata.txt", sep=None, engine="python")
meta["image_id"] = meta["filename_year"].astype(str)
q2 = q.merge(meta[["image_id", "length", "sightings"]], on="image_id", how="left")
g2 = g.merge(meta[["image_id", "length"]], on="image_id", how="left")
q2["year"] = pd.to_datetime(q2["date"]).dt.year
q2["maxsim"] = mx
known = q2["is_known"].to_numpy()

print("== max-sim by year and type ==")
for y in sorted(q2["year"].unique()):
    sel = q2["year"] == y
    k = q2[sel & known]["maxsim"]; n = q2[sel & ~known]["maxsim"]
    print("year %d known n=%d mean=%.4f | novel n=%d mean=%.4f" % (y, len(k), k.mean() if len(k) else float("nan"), len(n), n.mean() if len(n) else float("nan")))

from scipy.stats import spearmanr
ok = q2["length"].notna().to_numpy()
print("== length effects ==")
print("spearman(maxsim, query length) all: rho=%.3f p=%.1e" % spearmanr(q2["maxsim"][ok], q2["length"][ok]))
for lab, sel in (("known", known & ok), ("novel", (~known) & ok)):
    r = spearmanr(q2["maxsim"][sel], q2["length"][sel])
    print("  %s: rho=%.3f p=%.1e | mean length=%.1f" % (lab, r.statistic, r.pvalue, q2["length"][sel].mean()))
glen = g2["length"].to_numpy()
dl = np.abs(q2["length"].to_numpy() - glen[am])
ok2 = ~np.isnan(dl)
print("spearman(maxsim, |len_q - len_matched_gallery|): rho=%.3f p=%.1e" % spearmanr(mx[ok2], dl[ok2]))
print("mean |len diff| to argmax-gallery: %.2f vs random-gallery baseline: %.2f" % (
    np.nanmean(dl), np.nanmean(np.abs(q2["length"].to_numpy()[:, None] - glen[np.random.default_rng(0).integers(0, len(glen), (len(q2), 1))]).ravel())))

print("== gallery-neighbor date structure ==")
gdate = pd.to_datetime(g["date"]).to_numpy()
qdate = pd.to_datetime(q2["date"]).to_numpy()
ddays = np.abs((qdate - gdate[am]) / np.timedelta64(1, "D"))
print("median |days query -> argmax gallery|: %.0f (known) / %.0f (novel)" % (
    np.median(ddays[known]), np.median(ddays[~known])))
rand_am = np.random.default_rng(1).integers(0, len(g), len(q2))
print("random-gallery baseline median |days|: %.0f" % np.median(np.abs((qdate - gdate[rand_am]) / np.timedelta64(1, "D"))))
