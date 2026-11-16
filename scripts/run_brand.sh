#!/bin/bash
# Full pipeline for one make: pull -> crop -> filter -> label check. Usage: run_brand.sh <make> <work_dir> <out_dir>
set -e
B=$1; W=$2; O=$3; S=$(dirname "$0")
python3 $S/01_pull_hf.py $W/dedup-0.7.jsonl $W/raw/$B $B
python3 $S/02_crop.py $W/raw/$B $W/crops/$B yolo26n.pt
python3 $S/05_filter_crops.py $W/crops/$B $O/$B
python3 $S/06_flag_label_errors.py $O/$B
cp $W/raw/$B/manifest.csv $O/$B/source_manifest.csv; cp $W/crops/$B/crop_log.csv $O/$B/
rm -rf $W/raw/$B $W/crops/$B
echo "DONE $B"
