# Web app for Vercel

Same pipeline as `demo/app.py`, rebuilt to fit a Vercel Python function (no PyTorch):

- `api/models/yolo26n.onnx`: YOLO26n vehicle detector (Ultralytics export, 640 px input)
- `api/models/clip_vitb32_visual_int8.onnx`: CLIP ViT-B/32 image encoder, int8-quantised (89 MB)
- `api/models/classifier_head.json`: the logistic-regression head from `demo/probe_12_models.joblib`
- `api/index.py`: FastAPI app with `POST /api/recognise` (multipart field `file`) and `GET /api/health`
- `public/index.html`: the page; it resizes photos to 1600 px in the browser before upload

Accuracy check on the hand-checked test sets (top-1): Commons 87.5% (int8) vs 87.0% (full-precision
ONNX); Openverse 52.6% vs 55.5%. Function bundle is about 260 MB, under Vercel's 500 MB Python limit.

## Deploy

Import the GitHub repo in Vercel and set **Root Directory** to `web`. No other settings are needed.

## Run locally

```
pip install -r requirements.txt uvicorn
uvicorn api.index:app --reload      # API only, at http://127.0.0.1:8000/api/health
```
