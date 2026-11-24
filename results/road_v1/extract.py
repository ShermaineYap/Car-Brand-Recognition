"""Detect + track cars in a video, save up to 5 crops per track (>=1.5 s apart, >=80 px wide)."""
import sys, subprocess, json, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/claude/car-brand-recognition/api")
import recogniser as R
vid, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(out, exist_ok=True)
W = 1280
info = subprocess.check_output(["ffprobe","-v","error","-select_streams","v","-show_entries","stream=width,height","-of","csv=p=0",vid]).decode().strip().split(",")
w0, h0 = int(info[0]), int(info[1]); H = round(h0 * W / w0 / 2) * 2
p = subprocess.Popen(["ffmpeg","-nostdin","-v","error","-i",vid,"-vf",f"fps={fps},scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"], stdout=subprocess.PIPE)
def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); i=max(0,x2-x1)*max(0,y2-y1)
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)
tracks=[]; fi=-1; nid=0
while True:
    buf = p.stdout.read(W*H*3)
    if len(buf) < W*H*3: break
    fi += 1
    im = Image.fromarray(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
    dets = [d for d in R.detect(im) if d[1] == "car"]
    used=set()
    for box, kind, score in dets:
        best=None; bi=0.25
        for tr in tracks:
            if id(tr) in used or fi - tr["last"] > 2: continue
            v=iou(tr["box"], box)
            if v>bi: bi=v; best=tr
        if best is None:
            nid+=1; best={"id":nid,"crops":[],"last":fi,"box":box,"lastsave":-99}; tracks.append(best)
        used.add(id(best)); best["box"]=box; best["last"]=fi
        x0,y0,x1,y1=[int(v) for v in box]; bw=x1-x0
        if bw>=80 and score>=0.5 and fi-best["lastsave"]>=1.5*fps and len(best["crops"])<5 and y1 < H*0.97:
            pad=int(0.04*bw); c=im.crop((max(0,x0-pad),max(0,y0-pad),min(W,x1+pad),min(H,y1+pad)))
            fn=f'{best["id"]:05d}_{len(best["crops"])}.jpg'; c.save(os.path.join(out,fn),quality=92)
            best["crops"].append({"f":fn,"t":round(fi/fps,2),"w":bw}); best["lastsave"]=fi
keep=[{"id":t["id"],"crops":t["crops"]} for t in tracks if t["crops"]]
json.dump(keep, open(os.path.join(out,"tracks.json"),"w"))
print(os.path.basename(vid), "frames", fi+1, "tracks with crops", len(keep), "crops", sum(len(t["crops"]) for t in keep), flush=True)
