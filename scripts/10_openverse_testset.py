"""Download a real-world test set from Openverse (openly licensed photos, mostly Flickr).

Searches Openverse for each model, downloads up to MAX images, and records title, creator,
licence and source page so the photos can be credited. Labels come from the search, so every
image must be checked by eye before it is used for testing.

Usage: python 10_openverse_testset.py <out_dir>
"""
import csv, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1]); MAX = 160
QUERIES = {
    "perodua_myvi": ["perodua myvi"], "perodua_axia": ["perodua axia"], "perodua_bezza": ["perodua bezza"],
    "proton_saga": ["proton saga"], "proton_persona": ["proton persona"], "proton_x50": ["proton x50"],
    "proton_x70": ["proton x70"], "toyota_vios": ["toyota vios"], "honda_city": ["honda city"],
    "honda_civic": ["honda civic"], "honda_hr-v": ["honda hr-v", "honda vezel"], "ford_ranger": ["ford ranger"],
}
UA = {"User-Agent": "CarModelDatasetStudentProject/0.1 (non-commercial university coursework)"}


def get(url, binary=False):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                b = r.read(); return b if binary else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(int(e.headers.get("Retry-After") or 10 * (attempt + 1))); continue
            raise
        except Exception:
            time.sleep(5)
    raise RuntimeError(url)


OUT.mkdir(parents=True, exist_ok=True)
f = open(OUT / "manifest.csv", "w", newline=""); w = csv.writer(f)
w.writerow(["file", "class", "query", "title", "creator", "licence", "licence_url", "source", "source_page", "image_url"])
for cls, qs in QUERIES.items():
    got, seen = 0, set()
    for q in qs:
        for page in range(1, 13):
            if got >= MAX:
                break
            d = get("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(
                {"q": q, "page": page, "page_size": 20, "mature": "false"}))
            res = d.get("results", [])
            for r in res:
                if got >= MAX or r["id"] in seen:
                    continue
                seen.add(r["id"])
                title = r.get("title") or ""
                words = q.split()[-1].replace("-", "")
                if words.lower() not in re.sub(r"[-\s]", "", title.lower() + " " + " ".join(t["name"] for t in r.get("tags") or [])).lower():
                    continue  # the model name must appear in the title or tags
                rel = f"{cls}/{r['source']}_{r['id'][:13]}.jpg"
                try:
                    (OUT / cls).mkdir(exist_ok=True)
                    (OUT / rel).write_bytes(get(r["thumbnail"] + "?full_size=true", binary=True))  # served by Openverse, up to 1024 px
                except Exception as e:
                    print("skip", r["url"], e, flush=True); continue
                w.writerow([rel, cls, q, title, r.get("creator"), r.get("license"), r.get("license_url"),
                            r.get("source"), r.get("foreign_landing_url"), r["url"]]); f.flush()
                got += 1
                time.sleep(0.3)
            if len(res) < 20:
                break
            time.sleep(1)
    print(cls, got, flush=True)
print("ALLDONE", flush=True)
