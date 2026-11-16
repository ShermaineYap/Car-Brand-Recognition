# Perodua car model dataset (first pass)

Built 15 Nov 2026. Keep this dataset private: the Hugging Face source states no licence
and the photos belong to their original publishers. Share code and results, not the images.

## What is here

| Path | Contents |
|---|---|
| `classes.csv` | The class list and image counts per model |
| `hf_perodua/clean/<model>/` | 1,119 car crops that passed every automatic check |
| `hf_perodua/review/` | 1,145 crops set aside, sorted by reason (see below) |
| `hf_perodua/manifest.csv` | One row per downloaded image: keyword, year, generation hint, status, original URL |
| `website_perodua/perodua_360_sprites.csv` | The 38 360-degree sprite sheets on perodua.com.my (1,910 frames) |
| `scripts/` | The scripts that produced all of this, in run order |

## How the Hugging Face images were cleaned

Source: `mesolitica/google-image-malaysian-vehicle-dedup`, which indexes
`malaysia-ai/crawl-google-image-malaysian-vehicle`. 4,714 images carry a Perodua keyword.

1. **YOLO crop** (`02_crop.py`): 2,450 dropped because no car was found (interiors, parts,
   spec sheets) or the car was too small. 2,264 crops kept.
2. **View filter** (`05_filter_crops.py`): 406 close-ups, interiors, open boots and adverts
   moved to `review/view/`; 496 near-duplicates to `review/duplicate/`; 133 odd-ones-out to
   `review/outlier/`.
3. **Label check** (`06_flag_label_errors.py`): 110 crops that look like a different Perodua
   model moved to `review/label_suspect/`, with the suggested model in the file name.

## What still needs doing by hand

- **Check `clean/`.** From the samples viewed, roughly one in six or seven crops is still
  wrong: another Perodua model, a non-Perodua car, or a leftover close-up. This is an
  eyeballed estimate, not a measured rate. Delete or move those.
- **Rescue from `review/label_suspect/` and `review/outlier/`.** Many are good photos filed
  under the wrong model; move them to the right folder.
- **Generations are mixed.** Folders are per model. `manifest.csv` has a generation hint taken
  from the search keyword's year, but the photo itself is sometimes an older generation.
- **Traz and QV-E have no photos** in the Hugging Face set (too new). **Viva** is included
  although Perodua no longer sells it, because it is still common on the road.
- **No test set yet.** Use your own photos for testing, not images from this folder.

## Perodua website images

The site's robots.txt disallows automated access to its image folder (`/assets/`), so these
were not downloaded automatically. Perodua's terms allow private, non-commercial use, so you
can save the sheets yourself in a browser:

1. Open each `url` in `perodua_360_sprites.csv` and save the image, keeping its file name.
2. Run `python scripts/04_slice_perodua_sprites.py website_perodua/perodua_360_sprites.csv <saved_folder> website_perodua/frames`

Each sheet holds 50 frames (52 for the Alza) of one colour rotating through 360 degrees.
The slicing script was tested on a made-up sheet only, not on a real Perodua file. Use these
frames for training only, and keep all frames of one sheet on the same side of any split.

## Wikimedia Commons

Not included. Commons rate-limited the cloud machine this was built on. Run it on your own
computer instead:

    python scripts/03_pull_commons.py commons "myvi=Perodua Myvi" "axia=Perodua Axia" "bezza=Perodua Bezza" "alza=Perodua Alza" "ativa=Perodua Ativa" "aruz=Perodua Aruz" "viva=Perodua Viva"

It writes a `manifest.csv` with author and licence for attribution. Then run `02_crop.py`
on the result. This script downloaded 29 files before the limit hit and has not completed a
full run.

## Re-running or adding another brand

    pip install huggingface_hub pyarrow pillow ultralytics open_clip_torch scikit-learn requests
    python scripts/01_pull_hf.py dedup-0.7.jsonl raw/hf Proton
    python scripts/02_crop.py raw/hf crops/hf yolo26n.pt
    python scripts/05_filter_crops.py crops/hf filtered/hf
    python scripts/06_flag_label_errors.py filtered/hf

`dedup-0.7.jsonl` is in the mesolitica dataset repo. That index lists 5,144 Proton and 1,280
MINI images.
