"""Crop the main vehicle out of every image with a COCO-pretrained YOLO detector.

Keeps the largest confident car/truck/bus box, pads it slightly, and drops images
where no vehicle is found (interiors, dashboards, brochures) or the crop is tiny.

Usage: python 02_crop.py <raw_dir> <crop_dir> [weights]
Expects <raw_dir>/<model>/<image>; writes <crop_dir>/<model>/<image> and crop_log.csv
"""
import csv, sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
WEIGHTS = sys.argv[3] if len(sys.argv) > 3 else "yolo11s.pt"
VEHICLES = [2, 5, 7]          # COCO: car, bus, truck (SUVs/MPVs are often called truck/bus)
MIN_CONF, MIN_SIDE, MIN_FRAC, PAD = 0.40, 160, 0.10, 0.04
MAX_SIDE = 800                # longest side of the saved crop

model = YOLO(WEIGHTS)
files = sorted(p for p in RAW.glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
log = []
for k in range(0, len(files), 32):
    batch = files[k:k + 32]
    for p, r in zip(batch, model.predict([str(b) for b in batch], classes=VEHICLES, conf=MIN_CONF,
                                         imgsz=640, verbose=False)):
        rel = f"{p.parent.name}/{p.stem}.jpg"
        H, W = r.orig_shape
        if len(r.boxes) == 0:
            log.append([rel, "no_vehicle", 0, "", 0]); continue
        xyxy = r.boxes.xyxy.tolist()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
        j = max(range(len(areas)), key=areas.__getitem__)
        x1, y1, x2, y2 = xyxy[j]
        conf = float(r.boxes.conf[j])
        frac = areas[j] / (W * H)
        if min(x2 - x1, y2 - y1) < MIN_SIDE:
            log.append([rel, "too_small", len(areas), round(conf, 3), round(frac, 3)]); continue
        if frac < MIN_FRAC:
            log.append([rel, "vehicle_too_minor", len(areas), round(conf, 3), round(frac, 3)]); continue
        px, py = PAD * (x2 - x1), PAD * (y2 - y1)
        box = (max(0, int(x1 - px)), max(0, int(y1 - py)), min(W, int(x2 + px)), min(H, int(y2 + py)))
        d = OUT / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        c = Image.open(p).convert("RGB").crop(box)
        c.thumbnail((MAX_SIDE, MAX_SIDE))
        c.save(d / f"{p.stem}.jpg", "JPEG", quality=90)
        log.append([rel, "ok", len(areas), round(conf, 3), round(frac, 3)])
    print(k + len(batch), "/", len(files), flush=True)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "crop_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "status", "vehicles_found", "conf", "box_fraction"])
    w.writerows(log)
