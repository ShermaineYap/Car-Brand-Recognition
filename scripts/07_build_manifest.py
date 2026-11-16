"""Merge the per-step logs of one make into a single manifest.csv (one row per downloaded image).

Usage: python 07_build_manifest.py <out_dir>/<make>
"""
import csv, os, sys
from pathlib import Path

D = Path(sys.argv[1])
rd = lambda n: {r["file"]: r for r in csv.DictReader(open(D / n))} if (D / n).exists() else {}
src, crop, filt, lab = rd("source_manifest.csv"), rd("crop_log.csv"), rd("filter_log.csv"), rd("label_check_log.csv")
rows = []
for f, m in src.items():
    status, loc = "crop:" + crop[f]["status"], ""
    if crop[f]["status"] == "ok":
        fs = filt[f]["status"]
        if fs != "ok":
            status, loc = "filter:" + fs, f"review/{fs.split(':')[0]}/{f}"
        elif f in lab and lab[f]["status"] != "ok":
            p = lab[f]["predicted"]
            status, loc = f"label_suspect:maybe-{p}", f"review/label_suspect/{m['model']}/maybe-{p}__{os.path.basename(f)}"
        else:
            status, loc = "clean", "clean/" + f
    if loc and not (D / loc).exists():
        raise SystemExit(f"missing {loc}")
    rows.append([f, m["make"], m["model"], m["year"], m["keyword"], status, loc, m["image_url"], m["page_url"]])
with open(D / "manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source_file", "make", "model", "keyword_year", "keyword", "status", "path_in_dataset",
                "original_image_url", "original_page_url"])
    w.writerows(rows)
for n in ("source_manifest.csv", "crop_log.csv", "filter_log.csv", "label_check_log.csv"):
    (D / n).unlink(missing_ok=True)
print(D.name, len(rows))
