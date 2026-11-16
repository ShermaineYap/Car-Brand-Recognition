# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3621 across 13 training classes reported as 13 models
- Held-out 20% of the crawl: **77.4%** top-1 (optimistic: same source as training)
- Openverse test set: **55.1%** top-1, **81.4%** top-3 over 623 photos (6 near-copies of training images removed)

| Class | Train crops | Openverse test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 182 | 57 | 19 (33%) | perodua_axia (32) |
| perodua_axia | 189 | 4 | 4 (100%) | - |
| perodua_bezza | 182 | 0 | 0 (0%) | - |
| proton_saga | 228 | 105 | 70 (67%) | other_car (32) |
| proton_persona | 181 | 52 | 9 (17%) | proton_saga (23) |
| proton_x50 | 51 | 1 | 0 (0%) | other_car (1) |
| proton_x70 | 179 | 1 | 1 (100%) | - |
| toyota_vios | 144 | 30 | 23 (77%) | proton_saga (4) |
| honda_city | 197 | 96 | 46 (48%) | proton_saga (32) |
| honda_civic | 197 | 106 | 50 (47%) | other_car (40) |
| honda_hr-v | 220 | 88 | 50 (57%) | other_car (27) |
| ford_ranger | 771 | 83 | 71 (86%) | other_car (11) |
| other_car | 900 | 0 | 0 (0%) | - |
