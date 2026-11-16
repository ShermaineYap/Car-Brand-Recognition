# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 3621 across 13 training classes reported as 13 models
- Held-out 20% of the crawl: **77.4%** top-1 (optimistic: same source as training)
- Commons test set: **86.5%** top-1, **97.6%** top-3 over 548 photos (6 near-copies of training images removed)

| Class | Train crops | Commons test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 182 | 41 | 30 (73%) | perodua_axia (5) |
| perodua_axia | 189 | 29 | 23 (79%) | proton_persona (3) |
| perodua_bezza | 182 | 42 | 37 (88%) | perodua_axia (2) |
| proton_saga | 228 | 20 | 18 (90%) | perodua_bezza (2) |
| proton_persona | 181 | 37 | 33 (89%) | proton_saga (3) |
| proton_x50 | 51 | 18 | 18 (100%) | - |
| proton_x70 | 179 | 24 | 21 (88%) | proton_x50 (3) |
| toyota_vios | 144 | 80 | 64 (80%) | honda_city (10) |
| honda_city | 197 | 69 | 59 (86%) | honda_civic (5) |
| honda_civic | 197 | 92 | 84 (91%) | honda_city (4) |
| honda_hr-v | 220 | 55 | 51 (93%) | proton_saga (1) |
| ford_ranger | 771 | 41 | 36 (88%) | other_car (5) |
| other_car | 900 | 0 | 0 (0%) | - |
