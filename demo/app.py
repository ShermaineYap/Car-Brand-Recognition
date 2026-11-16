"""Car brand and model recognition demo.

Finds every car in a photo with YOLO, then names each one (for example "Perodua Myvi") with the
trained classifier. Opens a small web page where you can upload a photo or use your webcam.

Run:
    pip install -r requirements.txt
    python app.py              # then open the link it prints (usually http://127.0.0.1:7860)

Classifier: uses the fine-tuned model (car_classifier.pt + car_classifier.json from the Colab
notebook) if both files are in this folder; otherwise the baseline (probe_12_models.joblib).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
VEHICLE_CLASSES = [2, 5, 7]       # COCO car, bus, truck (YOLO often calls SUVs and pickups "truck")
DETECT_CONF = 0.35
MIN_BOX = 60                      # ignore cars smaller than this many pixels
MIN_AREA = 0.03                   # only name cars covering at least 3% of the photo; smaller ones get a thin grey box
BRANDS = {"perodua": "Perodua", "proton": "Proton", "toyota": "Toyota", "honda": "Honda", "ford": "Ford"}
COLOURS = [(46, 125, 50), (21, 101, 192), (198, 40, 40), (106, 27, 154), (239, 108, 0), (0, 131, 143)]


def pretty(label):
    """'honda_hr-v' -> ('Honda', 'HR-V'); 'other_car' -> ('Other', 'car')"""
    if label == "other_car":
        return "Other", "car"
    make, model = label.split("_", 1)
    nice = {"hr-v": "HR-V", "x50": "X50", "x70": "X70"}.get(model, model.replace("_", " ").title())
    return BRANDS.get(make, make.title()), nice


class FineTuned:
    """Model trained in the Colab notebook (timm)."""

    def __init__(self, weights, meta):
        import timm
        from torchvision import transforms as T
        m = json.load(open(meta))
        self.labels_of_class = m["labels"]
        self.labels = sorted(set(self.labels_of_class))
        self.net = timm.create_model(m["model_name"], pretrained=False, num_classes=len(m["classes"]))
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()
        s = m["img_size"]
        self.tf = T.Compose([T.Resize(s), T.CenterCrop(s), T.ToTensor(), T.Normalize(m["mean"], m["std"])])
        self.name = f"fine-tuned {m['model_name']}"
        self.unsure_below = 0.50   # below this confidence the guess is shown in grey with a "?"

    @torch.no_grad()
    def __call__(self, crops):
        p = self.net(torch.stack([self.tf(c) for c in crops])).softmax(-1).numpy()
        return _to_labels(p, self.labels_of_class, self.labels)


class Baseline:
    """Frozen CLIP features + logistic regression (probe_12_models.joblib)."""

    def __init__(self, path):
        import joblib, open_clip
        d = joblib.load(path)
        self.probe = d["probe"]
        self.labels_of_class = d.get("labels", d["classes"])
        self.labels = list(dict.fromkeys(self.labels_of_class))
        self.clip, _, self.prep = open_clip.create_model_and_transforms(d["arch"], pretrained=d["weights"])
        self.clip.eval()
        self.name = f"baseline (CLIP {d['arch']} + logistic regression)"
        self.unsure_below = 0.30   # its probabilities are flatter; at 0.30 it keeps 94% of cars at ~90% accuracy

    @torch.no_grad()
    def __call__(self, crops):
        e = self.clip.encode_image(torch.stack([self.prep(c) for c in crops]))
        e = (e / e.norm(dim=-1, keepdim=True)).numpy()
        return _to_labels(self.probe.predict_proba(e), self.labels_of_class, self.labels)


def _to_labels(p, labels_of_class, labels):
    """Sum class probabilities per reported model (e.g. myvi + myvi_old -> Myvi)."""
    M = np.zeros((len(labels_of_class), len(labels)))
    for i, l in enumerate(labels_of_class):
        M[i, labels.index(l)] = 1
    return p @ M, labels


def load_classifier():
    pt, js, jl = HERE / "car_classifier.pt", HERE / "car_classifier.json", HERE / "probe_12_models.joblib"
    if pt.exists() and js.exists():
        return FineTuned(pt, js)
    if jl.exists():
        return Baseline(jl)
    sys.exit("No classifier found: put car_classifier.pt + .json or probe_12_models.joblib next to app.py")


detector = YOLO("yolo26n.pt")      # downloaded automatically the first time
classifier = load_classifier()
print("Classifier:", classifier.name)


def recognise(image):
    """image: RGB numpy array. Returns (annotated image, table rows, summary text)."""
    if image is None:
        return None, [], "Upload a photo first."
    rgb = np.ascontiguousarray(image[..., :3])
    H, W = rgb.shape[:2]
    res = detector.predict(rgb[..., ::-1], classes=VEHICLE_CLASSES, conf=DETECT_CONF,
                           agnostic_nms=True, verbose=False)[0]   # agnostic: one box per car even if called car and truck
    boxes = [b for b in res.boxes.xyxy.tolist() if min(b[2] - b[0], b[3] - b[1]) >= MIN_BOX]
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    if not boxes:
        return rgb, [], "No car found. Try a photo where the car is larger and fully visible."
    out = rgb.copy()
    thick = max(2, round(min(H, W) / 300))
    small = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < MIN_AREA * H * W]
    boxes = [b for b in boxes if b not in small]
    for x1, y1, x2, y2 in small:                     # background cars: box only, no name
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (170, 170, 170), 1)
    if not boxes:
        return out, [], "Only small or distant cars found. Try a photo where the car fills more of the frame."
    crops = []
    for x1, y1, x2, y2 in boxes:
        px, py = 0.04 * (x2 - x1), 0.04 * (y2 - y1)
        crops.append(Image.fromarray(rgb).crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))))
    P, labels = classifier(crops)
    rows = []
    for k, ((x1, y1, x2, y2), p) in enumerate(zip(boxes, P), 1):
        order = np.argsort(-p)
        brand, model = pretty(labels[order[0]])
        conf = float(p[order[0]])
        sure = conf >= classifier.unsure_below and labels[order[0]] != "other_car"
        if labels[order[0]] == "other_car":
            name = "Other car"                         # a make/model outside the list it knows
        else:
            name = f"{brand} {model}" + ("" if sure else "?")
        top3 = ", ".join(f"{' '.join(pretty(labels[j]))} {p[j]:.0%}" for j in order[:3])
        if labels[order[0]] == "other_car":
            rows.append([k, "Other", "not one of the models it knows", f"{conf:.0%}", top3])
        else:
            rows.append([k, brand, model + ("" if sure else " (not sure)"), f"{conf:.0%}", top3])
        col = COLOURS[(k - 1) % len(COLOURS)] if sure else (120, 120, 120)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)
        text = f"{k}. {name} {conf:.0%}"
        scale = max(0.5, min(H, W) / 900)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = int(y1) - 6 if y1 > th + 12 else int(y1) + th + 8
        cv2.rectangle(out, (int(x1), ty - th - 6), (int(x1) + tw + 8, ty + 4), col, -1)
        cv2.putText(out, text, (int(x1) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    named = [f"{r[1]} {r[2]}" for r in rows if r[1] != "Other" and "not sure" not in r[2]]
    summary = f"Found {len(rows)} car(s)" + (": " + ", ".join(named) if named else "") + "."
    return out, rows, summary


def build_ui():
    import gradio as gr
    known = ", ".join(" ".join(pretty(l)) for l in classifier.labels if l != "other_car")
    examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))
    with gr.Blocks(title="Car brand and model recognition") as ui:
        gr.Markdown(f"# Car brand and model recognition\nUpload a photo (or use your webcam). "
                    f"Each car is boxed and named.\n\n**Models it knows:** {known}.  \n"
                    f"Other makes and models are labelled *Other car*. **Classifier:** {classifier.name}. "
                    f"Cars it is unsure about are shown in grey with a question mark.")
        with gr.Row():
            inp = gr.Image(label="Photo", type="numpy", sources=["upload", "webcam", "clipboard"])
            out = gr.Image(label="Result")
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["Car", "Brand", "Model", "Confidence", "Top 3 guesses"], interactive=False)
        btn = gr.Button("Recognise", variant="primary")
        btn.click(recognise, inp, [out, table, summary])
        inp.upload(recognise, inp, [out, table, summary])
        if examples:
            gr.Examples(examples, inp)
    return ui


if __name__ == "__main__":
    build_ui().launch()
