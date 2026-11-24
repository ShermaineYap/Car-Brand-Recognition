"""Retrain the classifier head with road-view crops added (weighted), and evaluate.
Usage: python train_road.py <road_weight> <label files...>   (labels: sheet index -> class, per sheet prefix)"""
import sys, json, csv, glob, os, collections
import numpy as np, torch, open_clip
from PIL import Image
from sklearn.linear_model import LogisticRegression
os.chdir("/home/claude/cars")
W_ROAD = float(sys.argv[1]); OUT = sys.argv[2]; LABS = sys.argv[3:]
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k"); model.eval(); torch.set_num_threads(2)
CACHE = "runs/v5/emb_cache.npz"; cache = dict(np.load(CACHE))
def prep(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w*s)), max(224, round(h*s))), Image.BICUBIC); w, h = im.size; l, t = (w-224)//2, (h-224)//2
    return ((np.asarray(im.crop((l, t, l+224, t+224)), np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)
def embed(paths):
    todo = [p for p in paths if p not in cache]
    for k in range(0, len(todo), 64):
        x = torch.from_numpy(np.stack([prep(p) for p in todo[k:k+64]]))
        with torch.no_grad(): e = model.encode_image(x); e = (e/e.norm(dim=-1, keepdim=True)).numpy()
        for p, v in zip(todo[k:k+64], e): cache[p] = v
    if todo: np.savez(CACHE, **cache)
    return np.stack([cache[p] for p in paths]) if paths else np.zeros((0, 512), np.float32)
MERGE = {"honda_city_hatchback": "honda_city"}
rows = list(csv.DictReader(open("meta/config_v5.csv")))
P, Y = [], []
for r in rows:
    for p in sorted(glob.glob(r["train_dir"] + "*.jpg")): P.append(p); Y.append(MERGE.get(r["class"], r["class"]))
# road crops: every crop of every labelled track
def track_crops(index_file, lab_file):
    idx = {str(x["k"]): x for x in json.load(open(index_file))}; lab = json.load(open(lab_file)); out = []
    for k, c in lab.items():
        it = idx[k]; d = f'/home/claude/road/crops/{it["video"]}'
        t = next(t for t in json.load(open(d + "/tracks.json")) if t["id"] == it["track"])
        out += [(f'{d}/{cc["f"]}', c, f'{it["video"]}#{it["track"]}') for cc in t["crops"]]
    return out
road = []
for spec in LABS:
    idxf, labf = spec.split(":"); road += track_crops(idxf, labf)
classes_crawl = set(Y)
extra_cls = collections.Counter(c for _, c, _ in road if c not in classes_crawl)
# add crawl folders for new classes when they exist (e.g. xpander), else require >=3 road tracks
for c in list(extra_cls):
    mk, md = c.split("_", 1); d = f"all/{mk}/clean/{md}/"
    fs = sorted(glob.glob(d + "*.jpg"))
    ntr = len({g for _, cc, g in road if cc == c})
    if fs or ntr >= 3:
        for p in fs: P.append(p); Y.append(c)
    else:
        road = [r for r in road if r[1] != c]
X = embed(P); XR = embed([p for p, _, _ in road])
classes = sorted(set(Y) | {c for _, c, _ in road}); ci = {c: i for i, c in enumerate(classes)}
y = np.array([ci[c] for c in Y]); yr = np.array([ci[c] for _, c, _ in road], dtype=int)
Xall = np.vstack([X, XR]); yall = np.concatenate([y, yr]); w = np.concatenate([np.ones(len(y)), np.full(len(yr), W_ROAD)])
clf = LogisticRegression(C=8.0, max_iter=4000).fit(Xall, yall, sample_weight=w)
print(f"crawl {len(y)}  road {len(yr)} crops from {len({g for *_, g in road})} tracks, classes {len(classes)}")
# --- test window: per-crop and per-track (mean prob) accuracy, and 'confident & right' at 0.30
tl = json.load(open("/home/claude/road/labels_test.json")); tidx = {str(x["k"]): x for x in json.load(open("/home/claude/road/sheets/test_index.json"))}
T = json.load(open("/home/claude/road/crops/TESTWIN/tracks.json"))
right = conf_right = conf_wrong = 0; crop_right = crop_n = 0
for k, c in tl.items():
    t = next(t for t in T if t["id"] == tidx[k]["track"]); fs = [f'/home/claude/road/crops/TESTWIN/{cc["f"]}' for cc in t["crops"]]
    pr = clf.predict_proba(embed(fs)); m = pr.mean(0); j = int(m.argmax())
    crop_right += int((pr.argmax(1) == ci.get(c, -1)).sum()); crop_n += len(fs)
    right += classes[j] == c; conf_right += (classes[j] == c and m[j] >= 0.30); conf_wrong += (classes[j] != c and m[j] >= 0.30)
    print(f"   {c:22s} -> {classes[j]:22s} {m[j]:.2f}")
print(f"TEST tracks {len(tl)}: top-1 {right}, shown&right {conf_right}, shown&WRONG {conf_wrong}; crops {crop_right}/{crop_n}")
def realworld(root):
    fs, lab = [], []
    for c in sorted(os.listdir(root)):
        for f in sorted(glob.glob(f"{root}/{c}/*.jpg")): fs.append(f); lab.append(c)
    pr = clf.predict_proba(embed(fs)); pred = np.array(classes)[pr.argmax(1)]
    return (pred == np.array(lab)).mean()
print(f"Commons {realworld('test_commons_v2'):.1%}  Openverse {realworld('test_openverse'):.1%}")
json.dump({"classes": classes, "labels": classes, "coef": clf.coef_.round(5).tolist(), "intercept": clf.intercept_.round(5).tolist(),
           "arch": "ViT-B-16", "weights": "laion2b_s34b_b88k"}, open(OUT, "w"))
