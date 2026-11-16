"""Flag likely wrong-model labels with cross-validated predictions on CLIP embeddings.

Trains a simple classifier on 5 folds and, for each image, checks what a model that never
saw it predicts. Images where the folder label gets a low probability and another model a
high one are moved to review/label_suspect/<folder label>/ with the suggested label in the log.

Usage: python 06_flag_label_errors.py <filtered_dir>     (the folder holding clean/ and review/)
"""
import csv, shutil, sys
from pathlib import Path

import numpy as np, open_clip, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(sys.argv[1])
model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()
files = sorted((ROOT / "clean").glob("*/*.jpg"))
counts = {}
for p in files:
    counts[p.parent.name] = counts.get(p.parent.name, 0) + 1
files = [p for p in files if counts[p.parent.name] >= 10]     # too few images to cross-validate
if len({p.parent.name for p in files}) < 2:
    print("fewer than two models with 10+ images; nothing to compare"); sys.exit(0)
classes = sorted({p.parent.name for p in files})
y = np.array([classes.index(p.parent.name) for p in files])
embs = []
with torch.no_grad():
    for k in range(0, len(files), 64):
        e = model.encode_image(torch.stack([prep(Image.open(p).convert("RGB")) for p in files[k:k + 64]]))
        embs.append((e / e.norm(dim=-1, keepdim=True)).numpy())
X = np.concatenate(embs)
P = cross_val_predict(LogisticRegression(C=10, max_iter=2000), X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
print("cross-validated accuracy on the noisy labels:", round(float((P.argmax(1) == y).mean()), 3))
rows = []
for i, p in enumerate(files):
    own, best = P[i, y[i]], P[i].argmax()
    suspect = best != y[i] and own < 0.15 and P[i, best] > 0.6
    rows.append([f"{p.parent.name}/{p.name}", "label_suspect" if suspect else "ok",
                 classes[best], round(float(own), 3), round(float(P[i, best]), 3)])
    if suspect:
        d = ROOT / "review" / "label_suspect" / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), d / f"maybe-{classes[best]}__{p.name}")
with open(ROOT / "label_check_log.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "status", "predicted", "p_folder_label", "p_predicted"]); w.writerows(rows)
print("flagged", sum(r[1] != "ok" for r in rows), "of", len(rows))
