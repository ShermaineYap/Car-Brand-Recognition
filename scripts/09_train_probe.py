"""Baseline classifier: frozen CLIP image features + logistic regression ("linear probe").

Trains on the cleaned crops, then scores on (a) a held-out 20% of the same crawl and
(b) an independent real-world test set (e.g. Openverse or Wikimedia Commons crops). Test images that are
near-copies of a training image are removed first, so the score is not inflated.

Usage: python 09_train_probe.py <config.csv> <test_crop_dir> <out_dir> [test set name]
config.csv columns: class, train_dir[, label]   (one row per class; train_dir holds that class's crops;
  optional label = the model name reported, so several classes such as myvi_gen3 and myvi_old can
  be trained separately and still count as "perodua_myvi")
test_crop_dir: <class>/<image>.jpg, class names matching config.csv
Writes results.md, confusion.png, predictions.csv and probe.joblib
"""
import csv, sys
from pathlib import Path

import joblib, numpy as np, open_clip, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

CFG, TEST, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
NAME = sys.argv[4] if len(sys.argv) > 4 else "Test set"
OUT.mkdir(parents=True, exist_ok=True)
ARCH, WEIGHTS = "ViT-B-32", "laion2b_s34b_b79k"
model, _, prep = open_clip.create_model_and_transforms(ARCH, pretrained=WEIGHTS)
model.eval()
torch.set_num_threads(max(1, torch.get_num_threads()))


CACHE = OUT / "embedding_cache.npz"
_cache = dict(np.load(CACHE)) if CACHE.exists() else {}


def embed(paths):
    todo = [p for p in paths if str(p) not in _cache]
    if todo:
        new = _embed(todo)
        _cache.update({str(p): e for p, e in zip(todo, new)})
        np.savez(CACHE, **_cache)
    return np.stack([_cache[str(p)] for p in paths])


def _embed(paths):
    out = []
    with torch.no_grad():
        for k in range(0, len(paths), 64):
            x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[k:k + 64]])
            e = model.encode_image(x)
            out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
            print(f"  {min(k + 64, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


rows = list(csv.DictReader(open(CFG)))
classes = [r["class"] for r in rows]
label_of = [r.get("label") or r["class"] for r in rows]          # class index -> reported model
labels = list(dict.fromkeys(label_of))
tr_paths, tr_y = [], []
for i, r in enumerate(rows):
    ps = sorted(Path(r["train_dir"]).glob("*.jpg"))
    tr_paths += ps; tr_y += [i] * len(ps)
te_paths, te_y = [], []
for i, c in enumerate(labels):
    ps = sorted((TEST / c).glob("*.jpg"))
    te_paths += ps; te_y += [i] * len(ps)
tr_y, te_y = np.array(tr_y), np.array(te_y)
print("embedding", len(tr_paths), "training crops"); Xtr = embed(tr_paths)
print("embedding", len(te_paths), "test crops"); Xte = embed(te_paths)

# drop test images that are near-copies of any training image
sim = Xte @ Xtr.T
leak = sim.max(1) >= 0.95
Xte, te_y = Xte[~leak], te_y[~leak]
te_paths = [p for p, l in zip(te_paths, leak) if not l]

clf = lambda: LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")

# (a) held-out slice of the crawl
a, b, ya, yb = train_test_split(Xtr, tr_y, test_size=0.2, stratify=tr_y, random_state=0)
lab = np.array([labels.index(l) for l in label_of])
acc_crawl = (lab[clf().fit(a, ya).predict(b)] == lab[yb]).mean()

# (b) independent real-world test set
probe = clf().fit(Xtr, tr_y)
Pc = probe.predict_proba(Xte)
M = np.zeros((len(classes), len(labels)))
for i, l in enumerate(label_of):
    M[i, labels.index(l)] = 1
P = Pc @ M                                    # sum class probabilities per reported model
pred = P.argmax(1)
top3 = (np.argsort(-P, 1)[:, :3] == te_y[:, None]).any(1).mean()
acc = (pred == te_y).mean()
joblib.dump({"classes": classes, "labels": label_of, "probe": probe, "arch": ARCH, "weights": WEIGHTS}, OUT / "probe.joblib")

with open(OUT / "predictions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "true", "predicted", "confidence", "correct"])
    for p, t, q, pr in zip(te_paths, te_y, pred, P):
        w.writerow([f"{p.parent.name}/{p.name}", labels[t], labels[q], round(float(pr[q]), 3), t == q])

cm = confusion_matrix(te_y, pred, labels=range(len(labels)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right"); ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 * cm[i].sum() else "#333")
ax.set_xlabel("Predicted"); ax.set_ylabel(f"True ({NAME})")
ax.set_title(f"{NAME}: {acc:.0%} top-1 over {len(te_y)} photos")
fig.tight_layout(); fig.savefig(OUT / "confusion.png", dpi=130)

lines = [f"# Baseline results ({ARCH} {WEIGHTS} features + logistic regression)", "",
         f"- Training crops: {len(tr_y)} across {len(classes)} training classes reported as {len(labels)} models",
         f"- Held-out 20% of the crawl: **{acc_crawl:.1%}** top-1 (optimistic: same source as training)",
         f"- {NAME}: **{acc:.1%}** top-1, **{top3:.1%}** top-3 over {len(te_y)} photos "
         f"({int(leak.sum())} near-copies of training images removed)", "",
         f"| Class | Train crops | {NAME} photos | Correct | Most common wrong guess |", "|---|---|---|---|---|"]
lab_tr = np.array([labels.index(l) for l in label_of])[tr_y]
for i, c in enumerate(labels):
    n = int((te_y == i).sum()); ok = int(cm[i, i])
    wrong = cm[i].copy(); wrong[i] = 0
    wg = f"{labels[wrong.argmax()]} ({wrong.max()})" if wrong.max() else "-"
    lines.append(f"| {c} | {(lab_tr == i).sum()} | {n} | {ok} ({ok / max(n, 1):.0%}) | {wg} |")
open(OUT / "results.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
