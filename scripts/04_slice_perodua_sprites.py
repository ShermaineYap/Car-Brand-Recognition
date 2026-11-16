"""Cut Perodua 360-degree sprite sheets into single frames.

The Perodua model pages load one sprite sheet per colour (a grid of frames).
Save the sheets listed in perodua_360_sprites.csv from your own browser into a folder,
keeping the file names, then run:

    python 04_slice_perodua_sprites.py <perodua_360_sprites.csv> <folder_with_saved_sheets> <out_dir>

Frames go to <out_dir>/<model>/<colour>_<frame>.jpg. Adjacent frames are near-duplicates:
keep every frame of one sheet on the same side of the train/validation split.
"""
import csv, math, sys
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
LIST, SRC, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
for r in csv.DictReader(open(LIST)):
    p = SRC / Path(r["sprite_file"]).name
    if not p.exists():
        print("missing", p.name); continue
    im = Image.open(p)
    frames, cols = int(r["frames"]), int(r["frames_per_row"])
    rows = math.ceil(frames / cols)
    fw, fh = im.width // cols, im.height // rows   # works even if the sheet was resized
    d = OUT / r["model"]
    d.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        box = ((i % cols) * fw, (i // cols) * fh, (i % cols + 1) * fw, (i // cols + 1) * fh)
        f = im.crop(box)
        if f.mode in ("RGBA", "LA", "P"):                  # flatten transparency onto white
            f = f.convert("RGBA"); bg = Image.new("RGB", f.size, "white"); bg.paste(f, mask=f.split()[3]); f = bg
        f.convert("RGB").save(d / f"{r['colour'].replace(' ', '')}_{i:02d}.jpg", quality=92)
    print(p.name, frames, "frames")
