import csv, glob, json, os, shutil, sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
os.chdir("/home/claude/cars")
OUT = "/home/claude/gpu/pkg"; shutil.rmtree(OUT, ignore_errors=True)
MERGE = {"honda_city_hatchback": "honda_city"}
jobs = []
for r in csv.DictReader(open("meta/config_v5.csv")):
    c = MERGE.get(r["class"], r["class"])
    for p in sorted(glob.glob(r["train_dir"] + "*.jpg")): jobs.append((p, f"train/{c}/{os.path.basename(p)}"))
for p in sorted(glob.glob("all/mitsubishi/clean/xpander/*.jpg")): jobs.append((p, f"train/mitsubishi_xpander/{os.path.basename(p)}"))
# road crops (labelled tracks)
def tracks(idxf, labf, tag):
    idx = {str(x["k"]): x for x in json.load(open(idxf))}; out = []
    for k, c in json.load(open(labf)).items():
        it = idx[k]; d = f'/home/claude/road/crops/{it["video"]}'
        t = next(t for t in json.load(open(d + "/tracks.json")) if t["id"] == it["track"])
        out.append((c, f'{tag}_{it["video"]}_{it["track"]}', [f'{d}/{cc["f"]}' for cc in t["crops"]]))
    return out
R = "/home/claude/road/"
road = tracks(R+"sheets/trA_index.json", R+"lab_trA.json", "A") + tracks(R+"sheets/trB_index.json", R+"lab_trB.json", "B")
test = tracks(R+"sheets/test_index.json", R+"labels_test.json", "T1")
test = [(c, g, [f.replace("/crops/TESTWIN/", "/crops/TESTWIN/") for f in fs]) for c, g, fs in test]
test += tracks(R+"sheets/testA_index.json", R+"labels_testA.json", "T2")
for c, g, fs in road:
    for i, f in enumerate(fs): jobs.append((f, f"road/{c}/{g}_{i}.jpg"))
man = []
for c, g, fs in test:
    for i, f in enumerate(fs): jobs.append((f, f"test_road/{g}_{i}.jpg")); man.append({"file": f"test_road/{g}_{i}.jpg", "track": g, "label": c})
for root in ["test_commons_v2", "test_openverse"]:
    for p in sorted(glob.glob(f"{root}/*/*.jpg")): jobs.append((p, f"{root}/{p.split('/')[-2]}/{os.path.basename(p)}"))
def do(j):
    src, dst = j; q = f"{OUT}/{dst}"; os.makedirs(os.path.dirname(q), exist_ok=True)
    im = Image.open(src).convert("RGB"); im.thumbnail((288, 288)); im.save(q, "JPEG", quality=88)
with ProcessPoolExecutor(2) as ex: list(ex.map(do, jobs, chunksize=200))
json.dump(man, open(f"{OUT}/test_road.json", "w"))
shutil.copy("/tmp/h_4.json", f"{OUT}/head_init.json")
print(len(jobs), "images;", len(road), "road tracks;", len(test), "test tracks")
