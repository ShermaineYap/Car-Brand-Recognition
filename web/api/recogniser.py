"""Car brand and model recognition without PyTorch: YOLO26n and CLIP ViT-B/32 in ONNX Runtime.

Same pipeline as demo/app.py: detect vehicles, crop each one, embed it with CLIP, then the
logistic-regression head picks one of 12 models or "other car".
"""
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path(__file__).parent / "models"
VEHICLE_CLASSES = (2, 5, 7)       # COCO car, bus, truck
DETECT_CONF = 0.35
NMS_IOU = 0.6
MIN_BOX = 60                      # px; smaller cars are ignored
MIN_AREA = 0.03                   # cars under 3% of the photo are background: box only, no name
UNSURE_BELOW = 0.30               # baseline probabilities are flat; 0.30 keeps 94% at ~90% accuracy
PAD = 0.04
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
BRANDS = {"perodua": "Perodua", "proton": "Proton", "toyota": "Toyota", "honda": "Honda", "ford": "Ford"}

_opts = ort.SessionOptions()
_opts.intra_op_num_threads = 2
_yolo = ort.InferenceSession(str(MODELS / "yolo26n.onnx"), _opts, providers=["CPUExecutionProvider"])
_clip = ort.InferenceSession(str(MODELS / "clip_vitb32_visual_int8.onnx"), _opts, providers=["CPUExecutionProvider"])
_head = json.load(open(MODELS / "classifier_head.json"))
_W = np.array(_head["coef"], dtype=np.float32).T
_b = np.array(_head["intercept"], dtype=np.float32)
_class_label = _head["labels"]                       # training class -> reported model
LABELS = list(dict.fromkeys(_class_label))
_M = np.zeros((len(_class_label), len(LABELS)), dtype=np.float32)
for i, l in enumerate(_class_label):
    _M[i, LABELS.index(l)] = 1


def pretty(label):
    """'honda_hr-v' -> ('Honda', 'HR-V'); 'other_car' -> ('Other', 'car')"""
    if label == "other_car":
        return "Other", "car"
    make, model = label.split("_", 1)
    nice = {"hr-v": "HR-V", "x50": "X50", "x70": "X70"}.get(model, model.replace("_", " ").title())
    return BRANDS.get(make, make.title()), nice


def _letterbox(im, size=640):
    w, h = im.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (dx, dy))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, dx, dy


def _nms(boxes, scores, iou):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ious = inter / (area(boxes[i:i + 1]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][ious < iou]
    return keep


def detect(im):
    """Vehicle boxes [x1, y1, x2, y2] in original pixels, largest first."""
    x, r, dx, dy = _letterbox(im)
    out = _yolo.run(None, {_yolo.get_inputs()[0].name: x})[0][0].T       # (8400, 84)
    scores = out[:, 4:][:, list(VEHICLE_CLASSES)].max(1)
    m = scores >= DETECT_CONF
    if not m.any():
        return []
    cx, cy, w, h = out[m, 0], out[m, 1], out[m, 2], out[m, 3]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b[:, [0, 2]] = (b[:, [0, 2]] - dx) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dy) / r
    W, H = im.size
    b = np.clip(b, 0, [W, H, W, H])
    keep = _nms(b, scores[m], NMS_IOU)                 # class-agnostic: one box per car
    boxes = [b[i].tolist() for i in keep]
    boxes = [bb for bb in boxes if min(bb[2] - bb[0], bb[3] - bb[1]) >= MIN_BOX]
    return sorted(boxes, key=lambda bb: -(bb[2] - bb[0]) * (bb[3] - bb[1]))


def _clip_input(crop):
    w, h = crop.size
    s = 224 / min(w, h)
    crop = crop.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = crop.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = np.asarray(crop.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
    return ((a - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


def classify(crops):
    """Probabilities per reported model for each crop."""
    x = np.stack([_clip_input(c) for c in crops]).astype(np.float32)
    e = _clip.run(None, {"pixel_values": x})[0]
    z = e @ _W + _b
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p @ _M


def recognise(image_bytes):
    im = Image.open(io.BytesIO(image_bytes))
    im = im.convert("RGB")
    W, H = im.size
    boxes = detect(im)
    small = [bb for bb in boxes if (bb[2] - bb[0]) * (bb[3] - bb[1]) < MIN_AREA * W * H]
    big = [bb for bb in boxes if bb not in small]
    cars = []
    if big:
        crops = []
        for x1, y1, x2, y2 in big:
            px, py = PAD * (x2 - x1), PAD * (y2 - y1)
            crops.append(im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))))
        for k, (bb, p) in enumerate(zip(big, classify(crops)), 1):
            order = np.argsort(-p)
            label = LABELS[order[0]]
            brand, model = pretty(label)
            conf = float(p[order[0]])
            cars.append({
                "n": k, "box": [round(v, 1) for v in bb], "label": label, "brand": brand, "model": model,
                "confidence": round(conf, 3), "other": label == "other_car",
                "sure": conf >= UNSURE_BELOW and label != "other_car",
                "top3": [{"name": " ".join(pretty(LABELS[j])), "p": round(float(p[j]), 3)} for j in order[:3]],
            })
    return {"width": W, "height": H, "cars": cars, "background_boxes": [[round(v, 1) for v in bb] for bb in small],
            "known_models": [" ".join(pretty(l)) for l in LABELS if l != "other_car"]}
