"""Make numbered contact sheets of each track's largest crop, with the current model's top guess."""
import sys, json, os, glob
sys.path.insert(0, "/home/claude/car-brand-recognition/api")
import recogniser as R
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
items = []
for d in sys.argv[2:]:
    for t in json.load(open(os.path.join(d, "tracks.json"))):
        best = max(t["crops"], key=lambda c: c["w"])
        if best["w"] < int(os.environ.get("MINW", "0")): continue
        items.append((os.path.basename(d), t["id"], os.path.join(d, best["f"]), len(t["crops"])))
ims = [Image.open(p).convert("RGB") for _, _, p, _ in items]
probs = []
for k in range(0, len(ims), 32):
    probs += list(R.classify(ims[k:k+32]))
labels = R.LABELS
rows = []
COLS, ROWS, TW, TH = 6, 5, 300, 220
for s in range(0, len(items), COLS*ROWS):
    o = Image.new("RGB", (COLS*TW, ROWS*TH), "white"); dr = ImageDraw.Draw(o)
    for j, k in enumerate(range(s, min(s+COLS*ROWS, len(items)))):
        im = ims[k].copy(); im.thumbnail((TW-6, TH-40))
        x, y = (j % COLS)*TW, (j//COLS)*TH
        o.paste(im, (x+3, y+36))
        p = probs[k]; i = int(p.argmax())
        dr.text((x+4, y+2), f"#{k}  {labels[i]} {p[i]:.2f}", fill="black", font=font)
        dr.text((x+4, y+18), f"{items[k][0][:14]} n={items[k][3]}", fill="gray", font=font)
        rows.append({"k": k, "video": items[k][0], "track": items[k][1], "guess": labels[i], "p": float(p[i])})
    o.save(f"{sys.argv[1]}_{s//(COLS*ROWS):02d}.jpg", quality=80)
json.dump(rows, open(sys.argv[1] + "_index.json", "w"))
print(len(items), "tracks ->", (len(items)+COLS*ROWS-1)//(COLS*ROWS), "sheets")
