# Conditional two-strike payoff test (DRAFT)

Do hitters who adjust their swing at two strikes suffer a smaller two-strike penalty? Per (batter, stand), 2024-25, ≥400 swings (**n=471**). Two estimators of the penalty: **COARSE FE** (`penalty_*`, within 3x3-location x pitch-group) and **MATCHED** (`matched_*`, each hitter's 2-strike swings vs his own early swings in the same exact pitch_type x Statcast zone — tighter control for the 2-strike pitch-mix shift). OLS across hitters, predictors standardized. **Positive `count_adj` on run-value penalty = adjusting pays off.** Observational draft — see caveats in the script.

Matched coverage: mean 76% of 2-strike swings matched (median 79%). League avg penalties — FE rv -0.0007, matched rv +0.0011 run value/swing at 2K (everyone drops).

## Outcome: penalty_rv  (COARSE FE, run value — higher = more resilient)  — n=471, R²=0.118
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |      0     | 0.043 |  0    | 1      |
| count_adj       |      0.162 | 0.044 |  3.7  | 0.0002 |
| swing_plus      |     -0.239 | 0.045 | -5.38 | 0      |
| repertoire_plus |     -0.156 | 0.044 | -3.53 | 0.0005 |
| logn            |      0.112 | 0.045 |  2.49 | 0.0132 |

## Outcome: matched_rv  (MATCHED, run value — higher = more resilient)  — n=471, R²=0.078
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |      0     | 0.044 |  0    | 1      |
| count_adj       |      0.123 | 0.045 |  2.74 | 0.0063 |
| swing_plus      |     -0.191 | 0.046 | -4.19 | 0      |
| repertoire_plus |     -0.143 | 0.045 | -3.18 | 0.0016 |
| logn            |      0.065 | 0.046 |  1.42 | 0.1576 |

## Outcome: penalty_whiff  (COARSE FE, whiff — lower = fewer extra whiffs)  — n=471, R²=0.181
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |      0     | 0.042 |  0    | 1      |
| count_adj       |     -0.409 | 0.042 | -9.69 | 0      |
| swing_plus      |     -0.117 | 0.043 | -2.72 | 0.0068 |
| repertoire_plus |     -0.08  | 0.043 | -1.87 | 0.0619 |
| logn            |     -0.052 | 0.043 | -1.21 | 0.2269 |

## Outcome: matched_whiff  (MATCHED, whiff — lower = fewer extra whiffs)  — n=471, R²=0.147
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |     -0     | 0.043 | -0    | 1      |
| count_adj       |     -0.377 | 0.043 | -8.74 | 0      |
| swing_plus      |     -0.07  | 0.044 | -1.61 | 0.1084 |
| repertoire_plus |     -0.078 | 0.043 | -1.8  | 0.0731 |
| logn            |      0.012 | 0.044 |  0.28 | 0.7829 |

## count_adj zero-order correlations with the two-strike penalties
- penalty_rv: +0.185
- matched_rv: +0.143
- penalty_whiff: -0.391
- matched_whiff: -0.368
