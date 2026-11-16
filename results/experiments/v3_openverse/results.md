# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3091 across 12 classes
- Held-out 20% of the crawl: **86.3%** top-1 (optimistic: same source as training)
- Openverse test set: **67.2%** top-1, **85.3%** top-3 over 516 photos (113 near-copies of training images removed)

| Class | Train crops | Openverse test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 273 | 20 | 19 (95%) | proton_persona (1) |
| perodua_axia | 189 | 4 | 3 (75%) | perodua_myvi (1) |
| perodua_bezza | 182 | 0 | 0 (0%) | - |
| proton_saga | 347 | 63 | 44 (70%) | proton_persona (15) |
| proton_persona | 223 | 32 | 28 (88%) | proton_saga (3) |
| proton_x50 | 51 | 1 | 0 (0%) | proton_persona (1) |
| proton_x70 | 179 | 1 | 1 (100%) | - |
| toyota_vios | 179 | 30 | 19 (63%) | proton_persona (5) |
| honda_city | 280 | 88 | 68 (77%) | proton_persona (10) |
| honda_civic | 197 | 106 | 38 (36%) | proton_saga (41) |
| honda_hr-v | 220 | 88 | 51 (58%) | proton_persona (15) |
| ford_ranger | 771 | 83 | 76 (92%) | proton_saga (3) |
