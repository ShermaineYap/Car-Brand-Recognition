"""Download a real-world test set from Wikimedia Commons (openly licensed photos).

Each model maps to one or more Commons categories chosen to match the generations in the
training data. Walks each category two levels down, skips race/police/concept/tuning
subcategories, samples up to MAX_PER_MODEL photos, and records author and licence.

Usage: python 08_commons_testset.py <out_dir> [categories.json] [max_per_model] [exclude_manifest.csv]
categories.json maps class -> list of Commons categories; exclude_manifest lists titles to skip (e.g. a test set).
"""
import csv, json, random, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1])
MAX_PER_MODEL = int(sys.argv[3]) if len(sys.argv) > 3 else 100
CATS = {
    "perodua_myvi": ["Perodua Myvi (third generation)"],
    "perodua_axia": ["Perodua Axia (B200)", "Perodua Axia (A300)"],
    "perodua_bezza": ["Perodua Bezza (2016)", "Perodua Bezza (2020)"],
    "proton_saga": ["Proton Saga (BT, Mk1–3)", "Proton Saga (BT, Mk4)"],
    "proton_persona": ["Proton Persona (BH)"],
    "proton_x50": ["Proton X50"],
    "proton_x70": ["Proton X70"],
    "toyota_vios": ["Toyota Vios (XP150)", "Toyota Vios (AC100)", "Toyota Vios GR Sport"],
    "honda_city": ["Honda City (7th generation)", "Honda City Hybrid"],
    "honda_civic": ["Honda Civic (2015)", "Honda Civic (2021)"],
    "honda_hr-v": ["Honda HR-V (2nd generation)", "Honda HR-V (3rd generation)", "Honda HR-V e:HEV"],
    "ford_ranger": ["Ford Ranger (International)"],
}
if len(sys.argv) > 2:
    CATS = json.load(open(sys.argv[2]))
EXCLUDE = {r["title"] for r in csv.DictReader(open(sys.argv[4]))} if len(sys.argv) > 4 else set()
SKIP = re.compile(r"competition|police|taxi|concept|tuning|interior|engine|ambulance|fire engine|military|mule|scale model|\btoys?\b|drawing|logo|badge", re.I)
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CarModelDatasetStudentProject/0.1 (non-commercial university coursework)"}


def fetch(url, binary=False):
    for attempt in range(12):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(int(e.headers.get("Retry-After") or 5 * (attempt + 1))); continue
            raise
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("gave up: " + url)


def api(**p):
    time.sleep(2.5)
    return fetch(API + "?" + urllib.parse.urlencode({**p, "format": "json"}))


def files_in(cat, depth=0, seen=None):
    seen = set() if seen is None else seen
    if cat in seen or depth > 1:
        return
    seen.add(cat)
    cont = {}
    while True:
        d = api(action="query", generator="categorymembers", gcmtitle="Category:" + cat, gcmtype="file",
                gcmlimit=200, prop="imageinfo", iiprop="url|extmetadata|mime", iiurlwidth=1024, **cont)
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if ii and ii.get("mime") in ("image/jpeg", "image/png") and not SKIP.search(p["title"]):
                yield cat, p["title"], ii
        if "continue" not in d:
            break
        cont = d["continue"]
    d = api(action="query", list="categorymembers", cmtitle="Category:" + cat, cmtype="subcat", cmlimit=200)
    for c in d.get("query", {}).get("categorymembers", []):
        name = c["title"].split(":", 1)[1]
        if not SKIP.search(name):
            yield from files_in(name, depth + 1, seen)


strip = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()
OUT.mkdir(parents=True, exist_ok=True)
man = OUT / "manifest.csv"
done = {r["file"] for r in csv.DictReader(open(man))} if man.exists() else set()
new = not man.exists()
f = open(man, "a", newline=""); w = csv.writer(f)
if new:
    w.writerow(["file", "class", "commons_category", "title", "author", "licence", "licence_url", "source_page"])
random.seed(0)
for cls, cats in CATS.items():
    cache = OUT / f"_listing_{cls}.json"          # category listing, so a restart skips the walk
    if cache.exists() and len(json.loads(cache.read_text())) > 0:
        items = [tuple(x) for x in json.loads(cache.read_text())]
    else:
        items, seen_titles = [], set()
        for c in cats:
            for sub, title, ii in files_in(c):
                if title not in seen_titles and title not in EXCLUDE:
                    seen_titles.add(title); items.append((sub, title, ii))
        random.shuffle(items)
        cache.write_text(json.dumps(items))
    got = 0
    for sub, title, ii in items[:MAX_PER_MODEL]:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", title.split(":", 1)[1])
        rel = f"{cls}/{Path(name).stem[:100]}.jpg"
        got += 1
        if rel in done or (OUT / rel).exists() and rel in done:
            continue
        (OUT / cls).mkdir(exist_ok=True)
        try:
            (OUT / rel).write_bytes(fetch(ii.get("thumburl") or ii["url"], binary=True))
        except Exception as e:
            print("skip", title, e, flush=True); continue
        m = ii.get("extmetadata", {})
        w.writerow([rel, cls, sub, title, strip(m.get("Artist", {}).get("value")),
                    m.get("LicenseShortName", {}).get("value", ""), m.get("LicenseUrl", {}).get("value", ""),
                    ii.get("descriptionurl", "")]); f.flush()
        time.sleep(0.5)
    print(cls, "found", len(items), "kept", min(got, MAX_PER_MODEL), flush=True)
print("ALLDONE", flush=True)
