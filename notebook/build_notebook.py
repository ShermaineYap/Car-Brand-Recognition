"""Builds finetune_car_classifier.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Fine-tune a car model classifier (Google Colab, free GPU)

Trains an image classifier on the cleaned car crops and scores it on the two hand-checked
real-world test sets (Wikimedia Commons and Openverse).

**Before you start**
1. In Colab: *Runtime → Change runtime type → T4 GPU*.
2. Upload `colab_data.zip` (from your `Car Brand Recognition/colab/` folder) to Google Drive, into a
   folder called `Car Brand Recognition`. Or skip Drive and upload it in step 2 when asked.
3. Run the cells from top to bottom (*Runtime → Run all*). Training takes about 10–20 minutes on a T4.

The baseline to beat (frozen CLIP features + logistic regression, same 13 training classes):
**86.5%** on Commons and **55.1%** on Openverse. The Openverse photos are mostly older generations,
which the baseline often calls *other car*.

Training classes: the 12 car models plus `other_car` (900 crops of other makes and models), so the
demo app can say "not one of the models I know". `train_v3` also has `*_old` folders (older
generations, counted as their model) and extra 2023 Vios photos."""),
md("## 1. Settings"),
code("""TRAIN_SET  = "train_v2"   # "train_v2" = cleaned crawl crops + other_car (best so far); "train_v3" = + older generations & 2023 Vios
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"   # strong, small; alternatives: "efficientnet_b2", "vit_base_patch16_clip_224.openai"
IMG_SIZE   = 224
EPOCHS     = 12
BATCH      = 32
LR         = 2e-4
SEED       = 0
DRIVE_DIR  = "/content/drive/MyDrive/Car Brand Recognition"   # where colab_data.zip lives and results are saved
QUICK_TEST = False        # True = tiny run to check the notebook works (no GPU needed)"""),
md("## 2. Get the data"),
code("""import os, zipfile, glob
os.makedirs("/content/data", exist_ok=True)
zip_path = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    if os.path.exists(f"{DRIVE_DIR}/colab_data.zip"):
        zip_path = f"{DRIVE_DIR}/colab_data.zip"
except Exception as e:
    print("Not in Colab or Drive not mounted:", e)
if zip_path is None and os.path.exists("colab_data.zip"):
    zip_path = "colab_data.zip"
if zip_path is None:
    from google.colab import files
    print("colab_data.zip not found on Drive, please upload it")
    zip_path = list(files.upload().keys())[0]
zipfile.ZipFile(zip_path).extractall("/content/data")
for d in sorted(glob.glob("/content/data/*")):
    print(d, len(glob.glob(d + "/*/*.jpg")), "images")"""),
md("## 3. Data loaders\nRandom crops, flips and colour changes during training; a balanced sampler so small classes are seen as often as big ones."),
code("""import json, random, numpy as np, torch, timm
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms as T

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

cfg = timm.data.resolve_data_config({}, model=MODEL_NAME)
mean, std = cfg["mean"], cfg["std"]
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.6)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.RandomGrayscale(0.05),
    T.ToTensor(), T.Normalize(mean, std),
    T.RandomErasing(p=0.2),
])
eval_tf = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(mean, std)])
# Note: Resize + CenterCrop can cut the ends off a wide car; padding to square is an alternative to try.

train_full = datasets.ImageFolder(f"/content/data/{TRAIN_SET}", transform=train_tf)
classes = train_full.classes
# label = reported model; folders ending in _old (older generations) count as their model
label_of = [c[:-4] if c.endswith("_old") else c for c in classes]
labels = sorted(set(label_of))
print(len(classes), "training classes ->", len(labels), "models")

# 10% of training crops held out to watch for overfitting (same source as training, so optimistic)
idx = list(range(len(train_full))); random.shuffle(idx)
n_val = len(idx) // 10
val_idx, tr_idx = idx[:n_val], idx[n_val:]
if QUICK_TEST:
    tr_idx, val_idx = tr_idx[:64], val_idx[:32]
val_set = Subset(datasets.ImageFolder(f"/content/data/{TRAIN_SET}", transform=eval_tf), val_idx)
tr_set = Subset(train_full, tr_idx)
counts = Counter(train_full.targets[i] for i in tr_idx)
weights = [1.0 / counts[train_full.targets[i]] for i in tr_idx]
sampler = WeightedRandomSampler(weights, num_samples=len(tr_idx), replacement=True)
nw = 2 if device == "cuda" else 0
train_dl = DataLoader(tr_set, batch_size=BATCH, sampler=sampler, num_workers=nw, drop_last=True)
val_dl = DataLoader(val_set, batch_size=64, num_workers=nw)

def test_loader(name):
    ds = datasets.ImageFolder(f"/content/data/{name}", transform=eval_tf)
    if QUICK_TEST:
        ds = Subset(ds, list(range(0, len(ds), max(1, len(ds) // 40))))
    return ds, DataLoader(ds, batch_size=64, num_workers=nw)

print("train", len(tr_set), "val", len(val_set))"""),
md("## 4. Model and training"),
code("""import time, math
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
epochs = 1 if QUICK_TEST else EPOCHS
steps = epochs * len(train_dl)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
M = torch.zeros(len(classes), len(labels))
for i, l in enumerate(label_of):
    M[i, labels.index(l)] = 1

@torch.no_grad()
def predict(dl):
    model.eval(); probs, ys = [], []
    for x, y in dl:
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            p = model(x.to(device)).float().softmax(-1).cpu()
        probs.append(p @ M); ys.append(y)
    return torch.cat(probs), torch.cat(ys)

best, history = 0, []
for ep in range(epochs):
    model.train(); t0 = time.time(); tot = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            loss = crit(model(x), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    P, Y = predict(val_dl)
    yl = torch.tensor([labels.index(label_of[int(v)]) for v in Y])
    acc = (P.argmax(1) == yl).float().mean().item()
    history.append((ep + 1, tot / len(train_dl), acc))
    print(f"epoch {ep+1}/{epochs}  loss {tot/len(train_dl):.3f}  held-out {acc:.1%}  ({time.time()-t0:.0f}s)")
    if acc >= best:
        best = acc; torch.save(model.state_dict(), "/content/best.pt")
model.load_state_dict(torch.load("/content/best.pt"))"""),
md("## 5. Score on the real-world test sets\nThese photos come from different websites than the training crops, so these are the honest numbers."),
code("""import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

results = {}
for name in ["test_commons", "test_openverse"]:
    ds, dl = test_loader(name)
    P, Y = predict(dl)
    base = ds.dataset if isinstance(ds, Subset) else ds
    yl = torch.tensor([labels.index(base.classes[int(v)]) for v in Y])
    top1 = (P.argmax(1) == yl).float().mean().item()
    top3 = (P.topk(3, 1).indices == yl[:, None]).any(1).float().mean().item()
    results[name] = (top1, top3, len(yl))
    print(f"{name}: top-1 {top1:.1%}  top-3 {top3:.1%}  over {len(yl)} photos")
    cm = confusion_matrix(yl, P.argmax(1), labels=range(len(labels)))
    for i, l in enumerate(labels):
        n = cm[i].sum()
        if n:
            w = cm[i].copy(); w[i] = 0
            print(f"   {l:16s} {cm[i,i]:3d}/{n:<3d} {cm[i,i]/n:5.0%}   most confused with {labels[w.argmax()] if w.max() else '-'}")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_title(f"{name}: {top1:.0%} top-1"); fig.tight_layout(); plt.show()"""),
md("## 6. Save the model for the demo app\nCopies `car_classifier.pt` and `car_classifier.json` to your Drive folder. Put both files in the `demo/` folder of the project and the app will use them instead of the baseline."),
code("""meta = {"model_name": MODEL_NAME, "img_size": IMG_SIZE, "mean": list(mean), "std": list(std),
        "classes": classes, "labels": label_of, "train_set": TRAIN_SET, "epochs": epochs,
        "results": {k: {"top1": v[0], "top3": v[1], "photos": v[2]} for k, v in results.items()},
        "history": history}
torch.save(model.state_dict(), "/content/car_classifier.pt")
json.dump(meta, open("/content/car_classifier.json", "w"), indent=1)
out = DRIVE_DIR if os.path.isdir(DRIVE_DIR) else "/content"
if out != "/content":
    import shutil
    shutil.copy("/content/car_classifier.pt", out); shutil.copy("/content/car_classifier.json", out)
print("saved to", out)
try:
    from google.colab import files
    files.download("/content/car_classifier.pt"); files.download("/content/car_classifier.json")
except Exception:
    pass"""),
md("""## Things to try next
- `TRAIN_SET = "train_v3"` to add older generations and the 2023 Vios. With the baseline this helped
  older-car photos slightly but cost about 2 points on Commons; a fine-tuned model may handle it better.
- A larger model (`convnext_small.fb_in22k_ft_in1k`) or `IMG_SIZE = 288`.
- More epochs if the held-out score is still rising at the end."""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nbf.write(nb, "finetune_car_classifier.ipynb")
print("written")
