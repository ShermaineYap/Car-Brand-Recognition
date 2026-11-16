# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3091 across 16 training classes reported as 12 models
- Held-out 20% of the crawl: **85.3%** top-1 (optimistic: same source as training)
- Openverse test set: **60.7%** top-1, **83.3%** top-3 over 516 photos (113 near-copies of training images removed)

| Class | Train crops | Openverse test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 273 | 20 | 20 (100%) | - |
| perodua_axia | 189 | 4 | 3 (75%) | perodua_myvi (1) |
| perodua_bezza | 182 | 0 | 0 (0%) | - |
| proton_saga | 347 | 63 | 27 (43%) | proton_persona (27) |
| proton_persona | 223 | 32 | 31 (97%) | proton_saga (1) |
| proton_x50 | 51 | 1 | 0 (0%) | honda_civic (1) |
| proton_x70 | 179 | 1 | 1 (100%) | - |
| toyota_vios | 179 | 30 | 16 (53%) | proton_persona (9) |
| honda_city | 280 | 88 | 72 (82%) | proton_persona (12) |
| honda_civic | 197 | 106 | 25 (24%) | proton_saga (41) |
| honda_hr-v | 220 | 88 | 48 (55%) | perodua_myvi (18) |
| ford_ranger | 771 | 83 | 70 (84%) | proton_persona (7) |
