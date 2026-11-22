"""Download openly licensed photos from Wikimedia Commons categories, with a licence manifest.

Usage: python 03_pull_commons.py <out_dir> <model>=<Category name> [...]
Walks each category and its subcategories (which are usually generations), downloads a
1280px version of each photo, and records author, licence and source page for attribution.
"""
import csv, os, re, sys, time
from pathlib import Path

import requests

OUT = Path(sys.argv[1])
API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "CarModelDatasetStudentProject/0.1 (non-commercial coursework; contact via Wikimedia user talk)"


def get(params):
    for attempt in range(15):
        r = S.get(API, params={**params, "format": "json"}, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(int(r.headers.get("Retry-After", 5 * (attempt + 1)))); continue
        return r.json()
    raise RuntimeError("API kept failing")


def walk(cat, depth=0, seen=None):
    """Yield (subcategory, file info) for every bitmap in cat and its subcategories."""
    seen = seen if seen is not None else set()
    if cat in seen or depth > 3:
        return
    seen.add(cat)
    cont = {}
    while True:
        d = get({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                 "gcmtype": "file", "gcmlimit": 200, "prop": "imageinfo",
                 "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 1280, **cont})
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if ii and ii.get("mime") in ("image/jpeg", "image/png"):
                yield cat, p["title"], ii
        if "continue" not in d:
            break
        cont = d["continue"]
        time.sleep(2)
    d = get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
             "cmtype": "subcat", "cmlimit": 200})
    for c in d["query"]["categorymembers"]:
        yield from walk(c["title"].split(":", 1)[1], depth + 1, seen)


strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
rows, done = [], set()
for spec in sys.argv[2:]:
    model, cat = spec.split("=", 1)
    n = 0
    for sub, title, ii in walk(cat):
        if title in done:
            continue
        if n >= int(os.environ.get("CAP", "100000")):
            break
        done.add(title)
        meta = ii.get("extmetadata", {})
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", title.split(":", 1)[1])
        name = Path(name).stem[:120] + ".jpg"
        dest = OUT / model / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        status = "ok"
        if not dest.exists():
            for attempt in range(5):
                r = S.get(ii.get("thumburl") or ii["url"], timeout=120)
                if r.status_code == 200:
                    dest.write_bytes(r.content); break
                time.sleep(int(r.headers.get("Retry-After", 5 * (attempt + 1))))
            else:
                status = f"http_{r.status_code}"
            time.sleep(1.5)
        rows.append([f"{model}/{name}", "perodua", model, sub, status, title,
                     strip(meta.get("Artist", {}).get("value")),
                     meta.get("LicenseShortName", {}).get("value", ""),
                     meta.get("LicenseUrl", {}).get("value", ""), ii.get("descriptionurl", "")])
        n += 1
    print(model, cat, n, flush=True)

with open(OUT / "manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "make", "model", "commons_category", "status", "title", "author", "licence",
                "licence_url", "source_page"])
    w.writerows(rows)
