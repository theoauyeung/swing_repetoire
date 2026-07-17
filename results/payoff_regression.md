# Adjustment-payoff regression (DRAFT)

Hitter-level OLS, MLB 2024-2025, batters with ≥400 tracked swings (**n = 450**). Predictors standardized (std-beta, comparable across terms). Production is local per-swing run value. Per-(batter, stand) metrics usage-weighted to batter level. **Observational, contemporaneous — association, not causal effect.** Caveats in `src/payoff_regression.py`.

Question: does `count_adj` (adjustability) carry a positive partial association with production after controlling for repertoire width (`repertoire_plus`), shape count (`k`), swing quality (`swing_plus`), and playing time (`logn`)?

## Outcome: rv_per_swing (mean delta_run_exp per swing)  — n=450, R²=0.328
| term            |   std_beta |    se |    t |      p |
|:----------------|-----------:|------:|-----:|-------:|
| intercept       |      0     | 0.039 | 0    | 1      |
| count_adj       |      0.036 | 0.039 | 0.92 | 0.3566 |
| repertoire_plus |      0.049 | 0.053 | 0.92 | 0.3556 |
| k               |      0.052 | 0.057 | 0.91 | 0.3659 |
| swing_plus      |      0.281 | 0.04  | 6.99 | 0      |
| logn            |      0.404 | 0.043 | 9.33 | 0      |

## Outcome: xwobacon (xwOBA on contact)  — n=450, R²=0.460
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |     -0     | 0.035 | -0    | 1      |
| count_adj       |     -0.05  | 0.035 | -1.41 | 0.1593 |
| repertoire_plus |      0.111 | 0.048 |  2.31 | 0.0211 |
| k               |      0.037 | 0.051 |  0.72 | 0.4733 |
| swing_plus      |      0.402 | 0.036 | 11.14 | 0      |
| logn            |      0.401 | 0.039 | 10.32 | 0      |

## count_adj zero-order correlations
- rv_per_swing: -0.015
- xwobacon: -0.116
- repertoire_plus: -0.029
- swing_plus: -0.119
- k: -0.012
