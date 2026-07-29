# Adjustability value — swing-level DML causal estimates

Swing-level DML (n~100k swings). Treatment = per-swing fitted situational shift magnitude from within-batter 5-fold cross-fitting, averaged across 3 dials (bat_speed, swing_length, swing_path_tilt) in within-batter SD units. Unsigned (absolute value). 2024-25, >=400 swings per (batter, stand).

**Method:** Robinson (1988) partial linear model. XGBoost nuisance models (GroupKFold on batter_id, 5 folds). Within-batter demeaning for batter FE. Clustered sandwich SE by batter.

**Confounders:** location surface (px, pz, px^2, pz^2, px*pz), count (balls, strikes), game state (base_state, outs_when_up), pitch group, platoon, pitcher quality. Batter-level traits (swing quality, repertoire, playing time, handedness) absorbed by demeaning.

**Two-strike subset:** strikes==2 filter applied before demeaning.

## Results

| treatment     | scope       | outcome               |      n |   theta |     se |   ci_lo |   ci_hi |     t |      p |   r2_T |   r2_Y |
|:--------------|:------------|:----------------------|-------:|--------:|-------:|--------:|--------:|------:|-------:|-------:|-------:|
| adjustability | Season-wide | Run value per swing   | 573196 |  0.0018 | 0.002  | -0.0021 |  0.0056 |  0.9  | 0.3699 |  0.486 |  0.021 |
| adjustability | Season-wide | Whiff rate            | 573198 |  0.0011 | 0.0031 | -0.0049 |  0.0072 |  0.37 | 0.7138 |  0.486 |  0.167 |
| adjustability | Two-strike  | Two-strike run value  | 217351 | -0.0042 | 0.0027 | -0.0094 |  0.001  | -1.58 | 0.1142 |  0.265 |  0.034 |
| adjustability | Two-strike  | Two-strike whiff rate | 217352 |  0.0014 | 0.0034 | -0.0053 |  0.008  |  0.4  | 0.6874 |  0.252 |  0.155 |
| adj_count     | Season-wide | Run value per swing   | 573196 |  0.0082 | 0.0025 |  0.0032 |  0.0132 |  3.23 | 0.0013 |  0.693 |  0.021 |
| adj_count     | Season-wide | Whiff rate            | 573198 | -0.0263 | 0.003  | -0.0321 | -0.0204 | -8.84 | 0      |  0.694 |  0.167 |
| adj_count     | Two-strike  | Two-strike run value  | 217351 |  0.0006 | 0.0032 | -0.0057 |  0.0068 |  0.18 | 0.8608 |  0.518 |  0.034 |
| adj_count     | Two-strike  | Two-strike whiff rate | 217352 | -0.0091 | 0.003  | -0.015  | -0.0032 | -3.02 | 0.0027 |  0.506 |  0.155 |
| adj_gamestate | Season-wide | Run value per swing   | 573196 | -0.0006 | 0.0019 | -0.0044 |  0.0032 | -0.29 | 0.7686 |  0.537 |  0.021 |
| adj_gamestate | Season-wide | Whiff rate            | 573198 | -0.0026 | 0.0019 | -0.0064 |  0.0012 | -1.35 | 0.1766 |  0.536 |  0.167 |
| adj_gamestate | Two-strike  | Two-strike run value  | 217351 | -0.0003 | 0.0031 | -0.0064 |  0.0059 | -0.09 | 0.9294 |  0.533 |  0.034 |
| adj_gamestate | Two-strike  | Two-strike whiff rate | 217352 | -0.008  | 0.0028 | -0.0136 | -0.0025 | -2.86 | 0.0045 |  0.534 |  0.155 |
| adj_pitch     | Season-wide | Run value per swing   | 573196 | -0.0031 | 0.0049 | -0.0128 |  0.0066 | -0.63 | 0.5311 |  0.917 |  0.021 |
| adj_pitch     | Season-wide | Whiff rate            | 573198 |  0.0194 | 0.0098 |  0.0002 |  0.0387 |  1.98 | 0.0484 |  0.918 |  0.167 |
| adj_pitch     | Two-strike  | Two-strike run value  | 217351 | -0.0006 | 0.0078 | -0.0158 |  0.0146 | -0.08 | 0.9392 |  0.921 |  0.034 |
| adj_pitch     | Two-strike  | Two-strike whiff rate | 217352 |  0.0171 | 0.0114 | -0.0052 |  0.0395 |  1.51 | 0.1328 |  0.921 |  0.155 |
| adj_platoon   | Season-wide | Run value per swing   | 573196 |  0.0019 | 0.0031 | -0.0042 |  0.008  |  0.62 | 0.5374 |  0.508 |  0.021 |
| adj_platoon   | Season-wide | Whiff rate            | 573198 | -0.001  | 0.0024 | -0.0058 |  0.0038 | -0.41 | 0.6798 |  0.532 |  0.167 |
| adj_platoon   | Two-strike  | Two-strike run value  | 217351 |  0.0021 | 0.0045 | -0.0067 |  0.0109 |  0.46 | 0.6447 |  0.535 |  0.034 |
| adj_platoon   | Two-strike  | Two-strike whiff rate | 217352 | -0.0087 | 0.0035 | -0.0155 | -0.0019 | -2.5  | 0.0126 |  0.547 |  0.155 |
