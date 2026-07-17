# Conditional two-strike payoff test (DRAFT)

Do hitters who adjust their swing at two strikes suffer a smaller two-strike penalty? Per (batter, stand), 2024-25, ≥400 swings (**n=471**). `penalty_rv` = within-location FE slope of run value (delta_run_exp) on two-strike (negative = production drops at 2K; less negative = resilient). `penalty_whiff` = same for whiff rate. OLS across hitters, predictors standardized. **Positive `count_adj` on penalty_rv = adjusting pays off.** Observational draft — see caveats in the script.

League average: penalty_rv = -0.0007 run value/swing at 2K (everyone drops), penalty_whiff = -0.0382.

## Outcome: penalty_rv  (higher (less negative) = more resilient)  — n=471, R²=0.118
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |      0     | 0.043 |  0    | 1      |
| count_adj       |      0.162 | 0.044 |  3.7  | 0.0002 |
| swing_plus      |     -0.239 | 0.045 | -5.38 | 0      |
| repertoire_plus |     -0.156 | 0.044 | -3.53 | 0.0005 |
| logn            |      0.112 | 0.045 |  2.49 | 0.0132 |

## Outcome: penalty_whiff  (lower = fewer extra whiffs at 2K)  — n=471, R²=0.181
| term            |   std_beta |    se |     t |      p |
|:----------------|-----------:|------:|------:|-------:|
| intercept       |      0     | 0.042 |  0    | 1      |
| count_adj       |     -0.409 | 0.042 | -9.69 | 0      |
| swing_plus      |     -0.117 | 0.043 | -2.72 | 0.0068 |
| repertoire_plus |     -0.08  | 0.043 | -1.87 | 0.0619 |
| logn            |     -0.052 | 0.043 | -1.21 | 0.2269 |

## count_adj zero-order correlations with the two-strike penalties
- penalty_rv: +0.185
- penalty_whiff: -0.391
