# Road-view retraining (v1)

The crawl photos are mostly clean listing shots, so the classifier struggled with dashcam views
(rear views, blur, partial occlusion). This adds hand-checked car crops from Malaysian dashcam videos.

- Training road data: 219 hand-labelled cars (421 crops, up to 5 per tracked car) from five CC BY
  dashcam drives (Penang George Town, Gelugor/Bayan Lepas, Gombak, a KL highway). Road crops weighted x4.
- Held-out test: 51 hand-labelled cars from a different KL traffic-jam video (never used for training).

| | Before | After |
|---|---|---|
| Top guess right (per car) | 13 / 51 | 29 / 51 |
| Shown (>= 0.30) and right | 5 | 22 |
| Shown (>= 0.30) but wrong | 5 | 8 |
| Commons photos (model top-1) | 79.8% | 81.0% |
| Openverse photos (model top-1) | 49.0% | 53.7% |

Perodua Viva labels were removed: from behind it was too easy to confuse with a 2nd-gen Myvi.
Files: extract.py (detect + track + crop), sheet.py (labelling sheets), train_road.py, label JSONs.
