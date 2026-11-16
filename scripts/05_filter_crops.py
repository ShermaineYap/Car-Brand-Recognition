"""Second cleaning pass on the YOLO crops, using CLIP image embeddings.

1. View filter: keeps crops that look like a whole car from outside; moves close-ups
   (lamps, wheels, mirrors), interiors, engine bays, open boots and adverts to review/.
2. Near-duplicate removal: drops crops that are almost identical to one already kept.
3. Outlier flag: within each model folder, moves the crops least similar to the rest
   to review/ (these are often a different car model).

Nothing is deleted: rejected crops go to <out>/review/<reason>/<model>/ for a manual look.

Usage: python 05_filter_crops.py <crop_dir> <out_dir>
"""
import csv, shutil, sys
from pathlib import Path

import open_clip, torch
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
GOOD = ["a photo of a whole car seen from the outside",
        "a photo of the front of a car", "a photo of the rear of a car", "a side view photo of a car"]
BAD = ["a close-up photo of a car headlight or tail light", "a close-up photo of a car wheel",
       "a close-up photo of a car door mirror or badge", "a photo of a car interior",
       "a photo of a car engine bay with the bonnet open", "a photo of an open car boot",
       "an advertisement or poster with large text", "a photo of several different cars together",
       "a photo of people standing in front of a car", "a drawing or cutaway diagram of a car"]
DUP_SIM, OUTLIER_FRAC = 0.96, 0.10

model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()
with torch.no_grad():
    T = model.encode_text(tok(GOOD + BAD)); T = T / T.norm(dim=-1, keepdim=True)

files = sorted(SRC.glob("*/*.jpg"))
embs = []
with torch.no_grad():
    for k in range(0, len(files), 64):
        x = torch.stack([prep(Image.open(p).convert("RGB")) for p in files[k:k + 64]])
        e = model.encode_image(x); embs.append(e / e.norm(dim=-1, keepdim=True))
        print(k + 64, "/", len(files), flush=True)
E = torch.cat(embs)
P = (100 * E @ T.T).softmax(-1)
good_p = P[:, :len(GOOD)].sum(1)
top = P.argmax(1)

status = {}
for i, p in enumerate(files):
    status[i] = "ok" if good_p[i] >= 0.5 else "view:" + (GOOD + BAD)[int(top[i])] if top[i] >= len(GOOD) else "view:unclear"

by_model = {}
for i, p in enumerate(files):
    if status[i] == "ok":
        by_model.setdefault(p.parent.name, []).append(i)

for m, idx in by_model.items():
    kept = []
    for i in idx:                                   # near-duplicates
        if kept and float((E[kept] @ E[i]).max()) >= DUP_SIM:
            status[i] = "duplicate"
        else:
            kept.append(i)
    c = E[kept].mean(0); c = c / c.norm()           # outliers vs the class centre
    sims = E[kept] @ c
    n_out = int(len(kept) * OUTLIER_FRAC)
    for j in sims.argsort()[:n_out].tolist():
        status[kept[j]] = "outlier"

rows = []
for i, p in enumerate(files):
    s = status[i]
    rel = f"{p.parent.name}/{p.name}"
    reason = s.split(":")[0]
    dest = OUT / "clean" / rel if s == "ok" else OUT / "review" / reason / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    rows.append([rel, s, round(float(good_p[i]), 3)])
with open(OUT / "filter_log.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "status", "whole_car_score"]); w.writerows(rows)
