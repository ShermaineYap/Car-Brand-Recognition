"""Car brand and model recognition without PyTorch: YOLO26s and CLIP ViT-B/16 in ONNX Runtime.

Detect vehicles, crop each one, embed it with CLIP ViT-B/16, then a logistic-regression head picks
one of ~180 make-model classes. When the model is unsure but the make is clear, report the make.
"""
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODELS = Path(__file__).parent / "models"
VEHICLE_CLASSES = (2, 3, 5, 7)    # COCO car, motorcycle, bus, truck
KIND = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DETECT_CONF = 0.30
NMS_IOU = 0.6
MIN_BOX = 24                      # px; smaller vehicles are ignored
MIN_NAME = 48                     # px; cars smaller than this are boxed as "car" but not named
MAX_NAMED = 16                    # name at most this many cars per image (largest first) to bound latency
UNSURE_BELOW = 0.30               # set from the hold-out: see docs/web-app.md
PAD = 0.04
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
BRANDS = {"perodua": "Perodua", "proton": "Proton", "toyota": "Toyota", "honda": "Honda", "ford": "Ford",
          "bmw": "BMW", "byd": "BYD", "mini": "MINI", "mercedes-benz": "Mercedes-Benz", "ora": "ORA",
          "mazda": "Mazda", "nissan": "Nissan", "lexus": "Lexus", "kia": "Kia", "isuzu": "Isuzu", "porsche": "Porsche",
          "mitsubishi": "Mitsubishi", "haval": "Haval", "chery": "Chery", "lamborghini": "Lamborghini", "tank": "Tank"}
MODEL_NAMES = {"hr-v": "HR-V", "x50": "X50", "x70": "X70", "x90": "X90", "s70": "S70", "cr-v": "CR-V", "br-v": "BR-V",
               "wr-v": "WR-V", "cx-3": "CX-3", "cx-30": "CX-30", "cx-5": "CX-5", "cx-60": "CX-60", "cx-8": "CX-8",
               "cx-9": "CX-9", "mx-5": "MX-5", "mx-30": "MX-30", "bt-50": "BT-50", "d-max": "D-Max", "mu-x": "MU-X",
               "x-trail": "X-Trail", "c-hr": "C-HR", "rav4": "RAV4", "gr86": "GR86", "86": "86", "gr_supra": "GR Supra",
               "gr_yaris": "GR Yaris", "gr_corolla": "GR Corolla", "fj_cruiser": "FJ Cruiser", "s-max": "S-Max",
               "ev6": "EV6", "atto_3": "Atto 3", "amg_gt": "AMG GT", "3_door": "3-Door", "5_door": "5-Door",
               "ix": "iX", "ix1": "iX1", "ix3": "iX3", "i3": "i3", "i4": "i4", "i5": "i5", "i7": "i7", "i8": "i8",
               "x1": "X1", "x2": "X2", "x3": "X3", "x4": "X4", "x5": "X5", "x6": "X6", "x7": "X7", "xm": "XM", "z4": "Z4",
               "m2": "M2", "m3": "M3", "m4": "M4", "m5": "M5", "h1": "H1", "h2": "H2", "h6": "H6", "h9": "H9",
               "es": "ES", "gs": "GS", "is": "IS", "lc": "LC", "lm": "LM", "ls": "LS", "lx": "LX", "nx": "NX",
               "rc": "RC", "rx": "RX", "ux": "UX", "rz": "RZ", "nv200": "NV200", "nv350": "NV350", "eqa": "EQA",
               "eqb": "EQB", "eqe": "EQE", "eqs": "EQS", "eqe_suv": "EQE SUV", "eqs_suv": "EQS SUV", "cla": "CLA",
               "cls": "CLS", "gla": "GLA", "glb": "GLB", "glc": "GLC", "gle": "GLE", "gls": "GLS", "sl": "SL",
               "slc": "SLC", "qq": "QQ", "718": "718", "911": "911", "500": "500", "v70": "V70"}
MAKE_SURE = 0.50                  # if the model is unsure but its make is this likely, report the make only

_opts = ort.SessionOptions()
_opts.intra_op_num_threads = 2
_yolo = ort.InferenceSession(str(MODELS / "yolo26s.onnx"), _opts, providers=["CPUExecutionProvider"])
YOLO_SIZE = _yolo.get_inputs()[0].shape[2]          # 960: small, distant vehicles need the extra pixels
def _load_half_weights(path):
    """The CLIP ViT-B/16 file stores its big weight tensors as float16 to halve its size;
    convert them back to float32 so inference matches full precision."""
    import onnx
    from onnx import numpy_helper
    parts = sorted(path.parent.glob(path.name + ".part*"))    # stored in <100 MB parts for GitHub
    data = b"".join(p.read_bytes() for p in parts) if parts else path.read_bytes()
    m = onnx.load_from_string(data)
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
    return ort.InferenceSession(m.SerializeToString(), _opts, providers=["CPUExecutionProvider"])


_clip = _load_half_weights(MODELS / "clip_vitb16_w16.onnx")
_head = json.load(open(MODELS / "classifier_head.json"))
_W = np.array(_head["coef"], dtype=np.float32).T
_b = np.array(_head["intercept"], dtype=np.float32)
_class_label = _head["labels"]                       # training class -> reported model
LABELS = list(dict.fromkeys(_class_label))
_M = np.zeros((len(_class_label), len(LABELS)), dtype=np.float32)
for i, l in enumerate(_class_label):
    _M[i, LABELS.index(l)] = 1
MAKES = sorted({l.split("_")[0] for l in LABELS if l != "other_car"})
_MK = np.zeros((len(LABELS), len(MAKES)), dtype=np.float32)
for i, l in enumerate(LABELS):
    if l != "other_car":
        _MK[i, MAKES.index(l.split("_")[0])] = 1


def pretty(label):
    """'honda_hr-v' -> ('Honda', 'HR-V'); 'other_car' -> ('Other', 'car')"""
    if label == "other_car":
        return "Other", "car"
    make, model = label.split("_", 1)
    nice = MODEL_NAMES.get(model, model.replace("_", " ").title().replace("-Class", "-Class"))
    return BRANDS.get(make, make.title()), nice


def _letterbox(im, size=None):
    size = size or YOLO_SIZE
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
    """Vehicles as (box [x1, y1, x2, y2] in original pixels, kind, score), largest first."""
    x, r, dx, dy = _letterbox(im)
    out = _yolo.run(None, {_yolo.get_inputs()[0].name: x})[0][0].T       # (anchors, 84)
    cls_scores = out[:, 4:][:, list(VEHICLE_CLASSES)]
    scores = cls_scores.max(1)
    m = scores >= DETECT_CONF
    if not m.any():
        return []
    kinds = np.array(VEHICLE_CLASSES)[cls_scores[m].argmax(1)]
    cx, cy, w, h = out[m, 0], out[m, 1], out[m, 2], out[m, 3]
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b[:, [0, 2]] = (b[:, [0, 2]] - dx) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dy) / r
    W, H = im.size
    b = np.clip(b, 0, [W, H, W, H])
    keep = _nms(b, scores[m], NMS_IOU)                 # class-agnostic: one box per vehicle
    dets = [(b[i].tolist(), KIND[int(kinds[i])], float(scores[m][i])) for i in keep]
    dets = [d for d in dets if min(d[0][2] - d[0][0], d[0][3] - d[0][1]) >= MIN_BOX]
    return sorted(dets, key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))


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
    dets = detect(im)
    nameable = [i for i, (bb, kind, _) in enumerate(dets)
                if kind != "motorcycle" and min(bb[2] - bb[0], bb[3] - bb[1]) >= MIN_NAME][:MAX_NAMED]
    probs = {}
    if nameable:
        crops = []
        for i in nameable:
            x1, y1, x2, y2 = dets[i][0]
            px, py = PAD * (x2 - x1), PAD * (y2 - y1)
            crops.append(im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))))
        probs = dict(zip(nameable, classify(crops)))
    cars = []
    for k, (bb, kind, det_score) in enumerate(dets):
        car = {"n": k + 1, "box": [round(v, 1) for v in bb], "kind": kind, "detect_score": round(det_score, 3)}
        if k not in probs:                             # motorcycles and very small cars: boxed, not named
            car.update({"named": False, "label": kind, "brand": "", "model": kind, "confidence": round(det_score, 3),
                        "other": False, "sure": False, "make_only": False, "make_confidence": 0.0, "top3": []})
            cars.append(car)
            continue
        p = probs[k]
        order = np.argsort(-p)
        label = LABELS[order[0]]
        brand, model = pretty(label)
        conf = float(p[order[0]])
        pm = p @ _MK
        make_i = int(pm.argmax())
        make_conf = float(pm[make_i])
        sure = conf >= UNSURE_BELOW and label != "other_car"
        make_only = (not sure) and label != "other_car" and make_conf >= MAKE_SURE
        if make_only:
            brand = BRANDS.get(MAKES[make_i], MAKES[make_i].title())
        car.update({
            "named": True, "label": label, "brand": brand, "model": model,
            "confidence": round(conf, 3), "other": label == "other_car",
            "sure": sure, "make_only": make_only, "make_confidence": round(make_conf, 3),
            "top3": [{"name": " ".join(pretty(LABELS[j])), "p": round(float(p[j]), 3)} for j in order[:3]],
        })
        cars.append(car)
    return {"width": W, "height": H, "cars": cars, "background_boxes": [],
            "known_models": [" ".join(pretty(l)) for l in LABELS if l != "other_car"],
            "known_makes": [BRANDS.get(m, m.title()) for m in MAKES]}
