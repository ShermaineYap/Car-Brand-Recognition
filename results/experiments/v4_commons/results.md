# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3091 across 16 training classes reported as 12 models
- Held-out 20% of the crawl: **85.3%** top-1 (optimistic: same source as training)
- Commons test set: **80.5%** top-1, **96.9%** top-3 over 548 photos (6 near-copies of training images removed)

| Class | Train crops | Commons test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 273 | 41 | 34 (83%) | proton_persona (3) |
| perodua_axia | 189 | 29 | 13 (45%) | perodua_myvi (12) |
| perodua_bezza | 182 | 42 | 35 (83%) | perodua_myvi (3) |
| proton_saga | 347 | 20 | 8 (40%) | proton_persona (7) |
| proton_persona | 223 | 37 | 29 (78%) | perodua_myvi (5) |
| proton_x50 | 51 | 18 | 17 (94%) | proton_x70 (1) |
| proton_x70 | 179 | 24 | 21 (88%) | proton_x50 (3) |
| toyota_vios | 179 | 80 | 61 (76%) | honda_city (13) |
| honda_city | 280 | 69 | 59 (86%) | honda_civic (4) |
| honda_civic | 197 | 92 | 78 (85%) | honda_city (12) |
| honda_hr-v | 220 | 55 | 49 (89%) | proton_x50 (2) |
| ford_ranger | 771 | 41 | 37 (90%) | proton_saga (4) |
