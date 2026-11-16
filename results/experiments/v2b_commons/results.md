# Baseline results (ViT-B-32 laion2b_s34b_b79k features + logistic regression)

- Training crops: 2721 across 12 classes
- Held-out 20% of the crawl: **85.9%** top-1 (optimistic: same source as training)
- Commons test set: **86.5%** top-1, **98.4%** top-3 over 548 photos (6 near-copies of training images removed)

| Class | Train crops | Commons test set photos | Correct | Most common wrong guess |
|---|---|---|---|---|
| perodua_myvi | 182 | 41 | 30 (73%) | perodua_axia (5) |
| perodua_axia | 189 | 29 | 23 (79%) | proton_persona (3) |
| perodua_bezza | 182 | 42 | 35 (83%) | perodua_axia (2) |
| proton_saga | 228 | 20 | 18 (90%) | perodua_bezza (2) |
| proton_persona | 181 | 37 | 32 (86%) | proton_saga (4) |
| proton_x50 | 51 | 18 | 17 (94%) | proton_x70 (1) |
| proton_x70 | 179 | 24 | 21 (88%) | proton_x50 (3) |
| toyota_vios | 144 | 80 | 64 (80%) | honda_city (9) |
| honda_city | 197 | 69 | 59 (86%) | honda_civic (6) |
| honda_civic | 197 | 92 | 84 (91%) | honda_city (4) |
| honda_hr-v | 220 | 55 | 50 (91%) | proton_saga (2) |
| ford_ranger | 771 | 41 | 41 (100%) | - |
