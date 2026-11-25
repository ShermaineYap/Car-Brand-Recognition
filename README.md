# Car Brand Recognition

Recognises the brand and model of cars common in Malaysia (for example *Perodua Myvi*) in a photo
or a dashcam video. Two stages: a COCO-pretrained YOLO finds each car, then an image classifier
names the cropped car.

**Live demo:** https://car-brand-recognition-three.vercel.app (photos and videos, 232 models from 29 makes; redeploys on every push to `main`)

## For IdeaLab Hacks judges

**The problem.** Two in three new cars sold in Malaysia are a Perodua or a Proton
([67% in the first half of 2026, MAA](https://paultan.org/2026/07/21/proton-and-perodua-gain-market-share-for-1h-2026-now-up-to-67-non-national-brands-left-with-33/)), and many of them
look alike: the Axia and Bezza share a front end, and the Bezza and Proton Saga are both small
sedans. Public car-recognition datasets such as Stanford Cars stop at 2012 models and contain no
Malaysian cars, so off-the-shelf models cannot name them.

**What it does.** Upload a photo or a dashcam clip. Every vehicle is boxed; each car big enough
to see is named with its make and model. When the classifier is not sure it falls back to the
make only, and when its top two guesses are close it says so (*Perodua Bezza or Axia?*) and
asks you to check the badge, instead of giving a confident wrong name.

**How well the live version works** (classifier `results/road_v1`, all test photos never used for training):

| Test | Result |
|---|---|
| Wikimedia Commons photos of 12 common Malaysian models | 81.0% named correctly |
| Openverse photos, mostly older generations | 53.7% |
| 51 cars in a KL traffic-jam dashcam video | 29 named correctly (13 before adding road crops) |

**What I learned**

- Clean listing photos do not prepare a model for the road. Adding 421 hand-labelled crops from
  five Creative Commons dashcam drives more than doubled the right answers on an unseen KL video (13 → 29 of 51).
- Cleaning labels helped more than splitting classes: removing 293 wrong training crops by hand raised the
  12-model classifier from 62.6% to 64.8% on the same 623 Openverse photos, while making older generations
  separate classes lowered Commons accuracy from 86.5% to 80.5%.
- Shrinking the model has a price: 8-bit quantisation to fit Vercel dropped 12-model accuracy from 92% to 78–81%,
  so the app keeps float16 weights split in two files instead.
- Confidence numbers are not answers. Probabilities for look-alikes such as Axia and Bezza can
  swap when a photo is resized, which is why close calls are now shown as two names.

**Built during the event.** All code, data work and results here were made between 15 and 25 Nov 2026,
inside the IdeaLab Hacks submission period (from 4 Nov). Solo project by Shermaine Yap.

**Tools and credits.** Ultralytics YOLO26 (detector), OpenCLIP ViT-B/16 (laion2b weights), scikit-learn,
ONNX Runtime, FastAPI, Vercel, Google Colab. Test and extra training photos from Wikimedia Commons
and Openverse, credited per photo in `data_manifests/`. The main training crawl
(`mesolitica/google-image-malaysian-vehicle-dedup` on Hugging Face) states no licence, so those
images are not shared here. AI coding assistants (Claude) helped write code and documentation.

**Limits.** Night photos, heavily modified cars, and models it does not know will still get a
name from its list or *car*; 99 of the crawl's classes had fewer than 40 clean photos. Photos and
video frames are processed and not stored.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ShermaineYap/Car-Brand-Recognition?quickstart=1)

## Run the demo on GitHub Codespaces

1. Click the badge above (or *Code → Codespaces → Create codespace on main*).
2. Wait for setup: the first time it installs PyTorch and the other packages (about 3–5 minutes).
3. The app starts by itself in the terminal and a browser tab opens. If no tab opens, go to the
   **Ports** tab and open port **7860**. The first photo takes longer while the models download.
4. Upload a photo or try the examples. Each car is boxed and named.

The codespace is private to you and stops after 30 minutes of inactivity. Stop or delete it from
github.com/codespaces when you are done, to save your free monthly hours. To share the demo with
someone, set port 7860 to *Public* in the Ports tab while the codespace is running (the link stops
working when the codespace stops). To run it on your own computer instead, see
[demo/README.md](demo/README.md).

## Current results

Top-1 accuracy on hand-checked real-world photos that were never used for training:

| Classifier | Commons test (548 photos, generations matched) | Openverse test (623 photos, mostly older cars) |
|---|---|---|
| Frozen CLIP ViT-B/32 + logistic regression, cleaned data + "other car" class | **86.5%** | 55.1% |

12 models: Perodua Myvi, Axia, Bezza; Proton Saga, Persona, X50, X70; Toyota Vios; Honda City,
Civic, HR-V; Ford Ranger. Plus an *other car* class. Full tables, confusion matrices and the
experiments that did not help are in [`results/`](results/) and summarised below.

| Version | Change | Commons | Openverse |
|---|---|---|---|
| v1 | first baseline on uncleaned crawl crops | 84.6%* | 62.6% |
| v2 | 293 wrong training crops removed by hand | 86.5% | 64.8% |
| v3 | + older generations and 2023 Vios, same classes | 84.7% | 67.2%** |
| v4 | older generations as separate classes | 80.5% | 60.7%** |
| v2 + other | v2 + 900-crop "other car" class (used by the demo) | 86.5% | 55.1% |

\* earlier, smaller Commons set (486 photos, only 12 Vios). \*\* on 516 photos; v2 scores 67.1% on the same 516.

## Repository layout

| Path | What |
|---|---|
| `scripts/` | Data pipeline and baseline training (run order below) |
| `notebook/finetune_car_classifier.ipynb` | Fine-tune a ConvNeXt/EfficientNet/ViT on a free Colab GPU and score both test sets |
| `api/`, `public/`, `vercel.json` | Vercel web app (ONNX models, no PyTorch); see [docs/web-app.md](docs/web-app.md) |
| `demo/` | Gradio web app: upload a photo, every car is boxed and named ([demo/README.md](demo/README.md)) |
| `results/` | Result tables, confusion matrices and per-photo predictions |
| `data_manifests/` | Class counts, removed-crop list, and credits (author, licence, link) for every Commons/Openverse photo |
| `docs/` | Data source research, dataset build notes |

## Data (not in this repo)

Images are kept out of git because of size and because the main training source (a Google Images
crawl on Hugging Face) states no licence. They live in the project folder on the author's computer:
`datasets/` (all brands), `commons_train/`, `test_commons/`, `results/test_sets/` and
`colab/colab_data.zip`. To rebuild them:

```
pip install huggingface_hub pyarrow pillow ultralytics open_clip_torch scikit-learn requests
# training crops from the crawl, one make at a time (needs dedup-0.7.jsonl from
# huggingface.co/datasets/mesolitica/google-image-malaysian-vehicle-dedup in <work_dir>)
bash scripts/run_brand.sh perodua <work_dir> <out_dir>
python scripts/07_build_manifest.py <out_dir>/perodua
# real-world test set and extra training photos from Wikimedia Commons (run on a home connection;
# Commons rate-limits cloud machines)
python scripts/08_commons_testset.py test_commons
python scripts/08_commons_testset.py commons_train scripts/commons_train_categories.json 250 test_commons/manifest.csv
python scripts/10_openverse_testset.py raw_test/openverse
# crop, then check every crop by eye before using it as a test photo
python scripts/02_crop.py <raw_dir> <crop_dir> yolo26n.pt
# baseline classifier
python scripts/09_train_probe.py scripts/config13_v2.csv <test_crop_dir> runs/v2 "Commons test set"
```

## Next steps

1. Run the Colab notebook to fine-tune; try `train_v3` (older generations) with the fine-tuned model.
2. Collect more photos for the weak pairs: Myvi vs Axia, Vios vs City, X70 vs X50.
3. Add more models (the crawl covers 223 make-model classes across 21 makes; see `data_manifests/crawl_classes.csv`).
