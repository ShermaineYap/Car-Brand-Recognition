# Car Brand Recognition

Recognises the brand and model of cars common in Malaysia (for example *Perodua Myvi*) in a photo.
Two stages: a COCO-pretrained YOLO finds each car, then an image classifier names the cropped car.

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
