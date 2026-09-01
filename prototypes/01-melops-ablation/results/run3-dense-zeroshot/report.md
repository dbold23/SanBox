# Melops Phase 1A ablation report

Backbone: `megadescriptor` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 151 | 411 | 291 | 0.153 | 0.392 | 0.279 | 0.397 |
| body | 151 | 411 | 291 | 0.119 | 0.307 | 0.225 | 0.438 |
| headless | 151 | 411 | 291 | 0.109 | 0.314 | 0.225 | 0.469 |
| cross_orientation | 80 | 162 | 381 | 0.074 | 0.290 | 0.185 | 0.748 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.
A decision floor gates all of it: if every crop arm is below 15
Rank-1 points the verdict is INCONCLUSIVE, because the kill rule
cannot express and the distributed rule is vacuous at that level.

**identity distributed - Approach 2 earns a hearing**

head - headless = 4.4 points; |headless - body| = 1.0 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2130 | 1.000 | 0.000 | 0.153 |
| 0.1 | 0.5326 | 0.876 | 0.069 | 0.141 |
| 0.2 | 0.5790 | 0.754 | 0.137 | 0.131 |
| 0.3 | 0.6076 | 0.637 | 0.213 | 0.117 |
| 0.4 | 0.6277 | 0.530 | 0.302 | 0.107 |
| 0.5 | 0.6473 | 0.433 | 0.405 | 0.097 |
| 0.6 | 0.6669 | 0.348 | 0.526 | 0.075 |
| 0.7 | 0.6823 | 0.258 | 0.639 | 0.066 |
| 0.8 | 0.7064 | 0.168 | 0.753 | 0.054 |
| 0.9 | 0.7375 | 0.073 | 0.859 | 0.015 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.3266 | 1.000 | 0.000 | 0.119 |
| 0.1 | 0.5081 | 0.886 | 0.082 | 0.109 |
| 0.2 | 0.5532 | 0.769 | 0.158 | 0.097 |
| 0.3 | 0.5911 | 0.662 | 0.247 | 0.085 |
| 0.4 | 0.6141 | 0.555 | 0.337 | 0.075 |
| 0.5 | 0.6393 | 0.462 | 0.447 | 0.061 |
| 0.6 | 0.6609 | 0.372 | 0.560 | 0.054 |
| 0.7 | 0.6806 | 0.268 | 0.653 | 0.046 |
| 0.8 | 0.7102 | 0.182 | 0.773 | 0.034 |
| 0.9 | 0.7417 | 0.092 | 0.887 | 0.022 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2781 | 1.000 | 0.000 | 0.109 |
| 0.1 | 0.4999 | 0.881 | 0.076 | 0.105 |
| 0.2 | 0.5471 | 0.783 | 0.179 | 0.097 |
| 0.3 | 0.5736 | 0.686 | 0.282 | 0.088 |
| 0.4 | 0.5987 | 0.574 | 0.364 | 0.080 |
| 0.5 | 0.6242 | 0.484 | 0.478 | 0.075 |
| 0.6 | 0.6478 | 0.392 | 0.588 | 0.056 |
| 0.7 | 0.6688 | 0.297 | 0.694 | 0.046 |
| 0.8 | 0.6894 | 0.192 | 0.787 | 0.027 |
| 0.9 | 0.7237 | 0.090 | 0.883 | 0.019 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1030 | 1.000 | 0.000 | 0.074 |
| 0.1 | 0.2093 | 1.000 | 0.144 | 0.074 |
| 0.2 | 0.2694 | 1.000 | 0.286 | 0.074 |
| 0.3 | 0.3448 | 1.000 | 0.428 | 0.074 |
| 0.4 | 0.4824 | 0.932 | 0.541 | 0.056 |
| 0.5 | 0.5588 | 0.753 | 0.606 | 0.049 |
| 0.6 | 0.5999 | 0.605 | 0.688 | 0.031 |
| 0.7 | 0.6374 | 0.481 | 0.777 | 0.025 |
| 0.8 | 0.6722 | 0.309 | 0.845 | 0.012 |
| 0.9 | 0.7069 | 0.148 | 0.919 | 0.006 |
