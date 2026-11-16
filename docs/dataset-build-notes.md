# Car make and model dataset, 20 more brands (first pass)

Built 15 Nov 2026 from the Hugging Face crawl `mesolitica/google-image-malaysian-vehicle-dedup`
(index of `malaysia-ai/crawl-google-image-malaysian-vehicle`). Perodua is in the separate
`perodua_dataset.zip`.

Keep the images private: the source states no licence and the photos belong to their original
publishers. Share code and results, not the images.

## Totals

46,482 images downloaded for 20 makes and 223 make-model classes:
17,065 clean crops, 9,576 set aside for review, 19,841 rejected (no car found or car too small).
`classes.csv` has the count for every class.

## Layout (one folder per make, split across the zips)

    <make>/clean/<model>/<year>_<shard>_<row>.jpg    crops that passed every automatic check
    <make>/review/view/<model>/                      close-ups, interiors, open boots, adverts
    <make>/review/duplicate/<model>/                 near-duplicates of a kept crop
    <make>/review/outlier/<model>/                   least typical crops of that model
    <make>/review/label_suspect/<model>/maybe-<x>__  look more like model <x> of the same make
    <make>/manifest.csv                              one row per downloaded image, with original URL

Crops are capped at 800 px on the longest side. The year in the file name comes from the search
keyword, not from the photo.

## Read this before training

- **The clean folders still contain wrong cars.** In the Perodua and Proton samples that were
  viewed, roughly one in six or seven crops was wrong. That is an eyeballed estimate, not a
  measured rate. For the other makes only nine crops from each of ten large classes were viewed;
  they showed the same kinds of error (for example a Ford F-150 and a Jeep under Ford Ranger,
  a BMW under Mercedes C-Class). Go through each folder once.
- **99 of the 223 classes have fewer than 40 clean crops** (column `fewer_than_40_clean` in
  `classes.csv`). Drop them, merge them, or collect more before training on them.
- **Classes are make + model, with generations and body styles merged** (for example every
  BMW 3 Series, or Mazda 3 sedan and hatchback together). Mercedes AMG variants sit under their
  base model. The mapping is in `scripts/models.py`.
- **One odd class:** `bmw_w` comes from a malformed keyword. Check it or delete it.
- **Not covered:** Jetour, Omoda Jaecoo (as a brand), Zeekr and iCaur are not in the source crawl.
  The Chery Omoda 5 is under Chery. Great Wall appears as Haval, ORA and Tank. Lamborghini has
  only 48 clean crops.
- **The label check only compares models within one make.** A BMW filed under Toyota would not
  be caught by it.
- **No test set.** Test on your own photos, not on images from these folders.

## Scripts (run order)

    pip install huggingface_hub pyarrow pillow ultralytics open_clip_torch scikit-learn requests
    bash scripts/run_brand.sh <make> <work_dir> <out_dir>      # work_dir must contain dedup-0.7.jsonl
    python scripts/07_build_manifest.py <out_dir>/<make>

`run_brand.sh` runs `01_pull_hf.py`, `02_crop.py`, `05_filter_crops.py` and
`06_flag_label_errors.py`. Makes available in the crawl but not yet pulled include Hyundai,
Volkswagen, Subaru, Volvo, Peugeot, Audi, Geely and Daihatsu.
