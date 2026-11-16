# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3091 across 12 classes
- Held-out 20% of the crawl: **86.3%** top-1 (optimistic: same source as training)
- Commons test set: **84.7%** top-1, **97.1%** top-3 over 548 photos (6 near-copies of training images removed)

| Class | Train crops | Commons test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 273 | 41 | 31 (76%) | proton_persona (5) |
| perodua_axia | 189 | 29 | 16 (55%) | perodua_myvi (10) |
| perodua_bezza | 182 | 42 | 36 (86%) | perodua_axia (2) |
| proton_saga | 347 | 20 | 13 (65%) | proton_persona (4) |
| proton_persona | 223 | 37 | 32 (86%) | perodua_myvi (5) |
| proton_x50 | 51 | 18 | 18 (100%) | - |
| proton_x70 | 179 | 24 | 21 (88%) | proton_x50 (3) |
| toyota_vios | 179 | 80 | 64 (80%) | honda_city (9) |
| honda_city | 280 | 69 | 61 (88%) | proton_x50 (3) |
| honda_civic | 197 | 92 | 80 (87%) | honda_city (9) |
| honda_hr-v | 220 | 55 | 51 (93%) | perodua_myvi (1) |
| ford_ranger | 771 | 41 | 41 (100%) | - |
