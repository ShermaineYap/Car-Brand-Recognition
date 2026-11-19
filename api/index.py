"""Vercel Python function: POST an image to /api/recognise, get the cars back as JSON."""
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, os.path.dirname(__file__))
from recogniser import LABELS, pretty, recognise  # noqa: E402

app = FastAPI(title="Car brand and model recognition")
MAX_BYTES = 4_000_000          # Vercel caps request bodies at 4.5 MB; the page shrinks photos first


@app.get("/api/health")
def health():
    return {"ok": True, "models": [" ".join(pretty(l)) for l in LABELS if l != "other_car"]}


@app.post("/api/recognise")
async def recognise_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Photo too large; please upload one under 4 MB")
    try:
        return recognise(data)
    except OSError:
        raise HTTPException(400, "That file is not an image the server can read (try JPEG or PNG)")
