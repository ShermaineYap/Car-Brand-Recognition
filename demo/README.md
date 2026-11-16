# Car brand and model recognition: demo app

Upload a photo (or use your webcam) and every car in it is boxed and named, for example
"Perodua Myvi 88%". Runs on your own computer; no GPU needed.

## Run it

```
cd demo
python3 -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
python app.py
```

Open the link it prints (usually http://127.0.0.1:7860). The first start takes a minute while
the models load. Photos in `examples/` are shown under the upload box for a quick try.

## What it does

1. **YOLO** (`yolo26n.pt`, pretrained on COCO) finds every car, SUV, pickup and bus.
2. Cars covering less than 3% of the photo get a thin grey box and no name (background cars).
3. Each remaining car is cropped and passed to the **classifier**, which picks one of 12 models,
   or *Other car* for makes and models it was not trained on.
4. Low-confidence guesses are drawn in grey with a "?".

## Classifier

- **Default:** `probe_12_models.joblib`, frozen CLIP ViT-B/32 features + logistic regression,
  trained on the cleaned crops plus an *other car* class (900 crops of other makes and models).
  Scores 86.5% top-1 on the 548-photo Commons test set (generations matched to training).
  It is weaker on older generations and labels many of them *Other car*.
- **Fine-tuned:** after running the Colab notebook (`notebook/finetune_car_classifier.ipynb`), copy `car_classifier.pt`
  and `car_classifier.json` into this folder. The app uses them automatically.

## Known limits

- It knows 12 models: Perodua Myvi, Axia, Bezza; Proton Saga, Persona, X50, X70; Toyota Vios;
  Honda City, Civic, HR-V; Ford Ranger. Anything else should come out as *Other car*, but it can
  still guess a similar-looking model (for example a Suzuki van called Bezza).
- Mix-ups seen in testing: Myvi vs Axia (the example photo `01` is one), Vios vs City, X70 vs X50.
- Pre-2016 generations are recognised less reliably.

## Example photo credits

All from Wikimedia Commons; see `examples/credits.csv` for title, author, licence and link.
