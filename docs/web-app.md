# Web app for Vercel

Files: `api/`, `public/`, `vercel.json` and `requirements.txt` at the repo root. Same pipeline as `demo/app.py`, rebuilt to fit a Vercel Python function (no PyTorch):

- `api/models/yolo26n.onnx`: YOLO26n vehicle detector (Ultralytics export, 640 px input)
- `api/models/clip_vitb16_w16.onnx.part0/1`: CLIP ViT-B/16 (laion2b_s34b_b88k) image encoder. Big weight tensors are
  stored as float16 (173 MB, split in two for GitHub) and converted back to float32 when the server starts.
  8-bit quantisation was tried and rejected: it dropped 12-model accuracy from 92% to 78–81%.
- `api/models/classifier_head.json`: logistic-regression head over 179 make-model classes from 21 makes
  (`scripts/11_train_big.py`, class list `scripts/config_big.csv`)
- `api/index.py`: FastAPI app with `POST /api/recognise` (multipart field `file`) and `GET /api/health`
- `public/index.html`: the page; it resizes photos to 1600 px in the browser before upload

Accuracy (top-1) of the 179-model classifier:

| Test | Model | Make |
|---|---|---|
| Grouped hold-out of the crawl (3,394 crops) | 59.7% | 79.8% |
| Commons, 12 Malaysian models (554 photos) | 80.1% | 89.4% |
| Openverse, older cars (629 photos) | 49.1% | 66.1% |

The page names a model when it is at least 30% confident (78% of those answers are right on the two
test sets) and otherwise shows only the make when that is at least 50% likely (92% right).

The 12-model classifier scored higher on Commons (87%) because it had fewer look-alikes to choose from.

## Video

The page samples a frame every 0.5 s (adjustable) in the browser, sends frames to `/api/recognise`
four at a time, links boxes across frames with an IoU tracker that never links across a scene cut,
and names each car by the label it got most often. The video file itself is never uploaded.

## Deploy

Import the GitHub repo in Vercel and leave Root Directory as `./`; Vercel detects it as a FastAPI app. Every push to `main` redeploys.

## Run locally

```
pip install -r requirements.txt uvicorn
uvicorn api.index:app --reload      # API only, at http://127.0.0.1:8000/api/health
```
