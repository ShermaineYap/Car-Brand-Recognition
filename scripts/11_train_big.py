"""Train the many-model classifier: CLIP ViT-B/16 (laion2b_s34b_b88k) image features
+ logistic regression over every make-model class with 25+ cleaned crops.

Evaluation:
  - grouped 20% hold-out of the crawl crops (split by source listing page, so near-duplicate
    photos from one listing never sit on both sides)
  - the hand-checked Commons and Openverse test sets (12 Malaysian models, real-world photos)
Writes classifier_head.json for the web app and results_big.md.

Usage: python 11_train_big.py <config.csv> <unused> <out_dir> [seed_cache.npz]
Classes in MERGE are folded into another class (their crawl folders were mostly the other model).
"""
import csv, json, os, sys, glob, collections
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

CFG, ONNX, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
import torch, open_clip
ARCH, PRETRAINED = "ViT-B-16", "laion2b_s34b_b88k"
_model, _, _prep = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
_model.eval(); torch.set_num_threads(2)
MERGE = {"honda_city_hatchback": "honda_city"}
CACHE = OUT / "emb_cache.npz"
cache = dict(np.load(CACHE)) if CACHE.exists() else {}
if len(sys.argv) > 4:                     # reuse embeddings computed earlier with the same model
    d = np.load(sys.argv[4])
    for k in ("X", "XT"):
        pass
    cache.update({k: v for k, v in d.items()} if "paths" not in d else {})


def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - 224) // 2, (h - 224) // 2
    return ((np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def embed(paths):
    todo = [p for p in paths if p not in cache]
    for k in range(0, len(todo), 64):
        x = torch.from_numpy(np.stack([prep(p) for p in todo[k:k + 64]]).astype(np.float32))
        with torch.no_grad():
            e = _model.encode_image(x); e = (e / e.norm(dim=-1, keepdim=True)).numpy()
        for p, v in zip(todo[k:k + 64], e):
            cache[p] = v
        if (k // 64) % 20 == 0:
            print(f"  embedded {k + len(todo[k:k + 64])}/{len(todo)}", flush=True)
            np.savez(CACHE, **cache)
    np.savez(CACHE, **cache)
    return np.stack([cache[p] for p in paths])


# group key = source page (crawl) so one listing's photos stay on one side of the split
pages = {}
for m in glob.glob("all/*/manifest.csv") + ["dataset/hf_perodua/manifest.csv"]:
    for r in csv.DictReader(open(m)):
        if r.get("path_in_dataset", "").startswith("clean/"):
            pages[os.path.basename(r["path_in_dataset"])] = r.get("original_page_url") or r["path_in_dataset"]

rows = list(csv.DictReader(open(CFG)))
classes = sorted({MERGE.get(r["class"], r["class"]) for r in rows})
P, y, g = [], [], []
for r in rows:
    i = classes.index(MERGE.get(r["class"], r["class"]))
    for p in sorted(glob.glob(r["train_dir"] + "*.jpg")):
        P.append(p); y.append(i); g.append(pages.get(os.path.basename(p), p))
y = np.array(y)
print(len(P), "crops", len(classes), "classes", flush=True)
X = embed(P)

rng = np.random.RandomState(0)
uniq = sorted(set(g)); rng.shuffle(uniq)
held = set(uniq[: len(uniq) // 5])
te = np.array([gi in held for gi in g])
clf = lambda: LogisticRegression(C=8.0, max_iter=4000)   # no class balancing: it hurt real-world accuracy
m = clf().fit(X[~te], y[~te])
pr = m.predict_proba(X[te]); yt = y[te]
top1 = (pr.argmax(1) == yt).mean(); top3 = (np.argsort(-pr, 1)[:, :3] == yt[:, None]).any(1).mean()
make = np.array([c.split("_")[0] for c in classes])
mk_acc = (make[pr.argmax(1)] == make[yt]).mean()
per = collections.defaultdict(list)
for t, p_ in zip(yt, pr.argmax(1)):
    per[classes[t]].append(t == p_)

# final model on all crops
final = clf().fit(X, y)
json.dump({"classes": classes, "labels": classes, "coef": final.coef_.round(5).tolist(),
           "intercept": final.intercept_.round(5).tolist(), "arch": ARCH, "weights": PRETRAINED},
          open(OUT / "classifier_head.json", "w"))

def realworld(root):
    fs, lab = [], []
    for c in sorted(os.listdir(root)):
        if os.path.isdir(f"{root}/{c}"):
            for f in sorted(glob.glob(f"{root}/{c}/*.jpg")):
                fs.append(f); lab.append(c)
    E = embed(fs); p = final.predict_proba(E)
    pred = np.array(classes)[p.argmax(1)]
    return (pred == np.array(lab)).mean(), (make[p.argmax(1)] == np.array([l.split("_")[0] for l in lab])).mean(), len(fs)

lines = ["# Many-model classifier", "",
         f"- {len(classes)} classes from {len(set(make))} makes, {len(P)} training crops",
         f"- Grouped hold-out of the crawl ({te.sum()} crops): model top-1 **{top1:.1%}**, top-3 {top3:.1%}, make {mk_acc:.1%}"]
for name, root in [("Commons (12 Malaysian models)", "test_commons_v2"), ("Openverse (older cars)", "test_openverse")]:
    a, b, n = realworld(root)
    lines.append(f"- {name}, {n} photos: model top-1 **{a:.1%}**, make {b:.1%}")
lines += ["", "| Class | Hold-out crops | Hold-out top-1 |", "|---|---|---|"]
for c in classes:
    v = per.get(c, [])
    lines.append(f"| {c} | {len(v)} | {np.mean(v):.0%} |" if v else f"| {c} | 0 | - |")
open(OUT / "results_big.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:7]))
