"""Pull images for one brand from the Malaysian vehicle crawl on Hugging Face.

Uses the dedup index (dedup-0.7.jsonl) to pick rows, and reads only the Parquet
row groups that contain them, so the full 100 GB crawl is never downloaded.

Usage: python 01_pull_hf.py <dedup-0.7.jsonl> <out_dir> [make]      (make as in models.py, e.g. toyota)
Writes <out_dir>/<model>/<year>_<shard>_<row>.jpg (longest side capped at 1280 px) and manifest.csv
"""
import csv, io, json, re, sys, collections, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models import parse

INDEX, OUT = Path(sys.argv[1]), Path(sys.argv[2])
BRAND = (sys.argv[3] if len(sys.argv) > 3 else "perodua").lower()
REPO = "datasets/malaysia-ai/crawl-google-image-malaysian-vehicle/data/"

rows = [json.loads(l) for l in open(INDEX)]
by_shard = collections.defaultdict(dict)  # shard -> {row index: keyword}
for r in rows:
    p = parse(r["keyword"])
    if p and p[0] == BRAND:
        for i in r["selected_indices"]:
            by_shard[r["filename"]][i] = r["keyword"]


def pull(shard):
    want = by_shard[shard]
    fs = HfFileSystem()
    out = []
    with fs.open(REPO + shard, "rb", block_size=4 * 2**20) as f:
        pf = pq.ParquetFile(f)
        start = 0
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            idx = [i for i in want if start <= i < start + n]
            if idx:
                tb = pf.read_row_group(g).to_pylist()
                for i in idx:
                    row = tb[i - start]
                    status = "ok"
                    if row["keyword"] != want[i]:
                        status = "keyword_mismatch"
                    _, model, year = parse(want[i])
                    model = model.replace(" ", "_")
                    name = f"{year}_{shard[6:11]}_{i:05d}.jpg"
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(row["parent_href"] or "").query)
                    if status == "ok":
                        try:
                            im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                            im.thumbnail((1280, 1280))
                            d = OUT / model
                            d.mkdir(parents=True, exist_ok=True)
                            im.save(d / name, "JPEG", quality=90)
                            w, h = im.size
                        except Exception as e:  # unreadable image
                            status, w, h = f"bad_image:{type(e).__name__}", 0, 0
                    else:
                        w = h = 0
                    out.append([f"{model}/{name}", BRAND, model, year, want[i], row["keyword"],
                                status, w, h, row["alt_text"],
                                q.get("imgurl", [""])[0], q.get("imgrefurl", [""])[0]])
            start += n
    print(shard, len(out), flush=True)
    return out


OUT.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(6) as ex:
    res = [r for part in ex.map(pull, sorted(by_shard)) for r in part]
with open(OUT / "manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "make", "model", "year", "keyword", "row_keyword", "status", "width", "height",
                "alt_text", "image_url", "page_url"])
    w.writerows(res)
print("total", len(res), collections.Counter(r[6].split(":")[0] for r in res))
