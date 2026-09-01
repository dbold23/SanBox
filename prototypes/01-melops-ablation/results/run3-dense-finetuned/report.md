# Melops Phase 1A ablation report

Backbone: `finetuned:/root/runs/ft-mega3/checkpoint.pt` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 58 | 154 | 136 | 0.182 | 0.591 | 0.366 | 0.409 |
| body | 58 | 154 | 136 | 0.292 | 0.591 | 0.439 | 0.655 |
| headless | 58 | 154 | 136 | 0.273 | 0.682 | 0.449 | 0.592 |
| cross_orientation | 34 | 61 | 134 | 0.213 | 0.508 | 0.362 | 0.540 |

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

head - headless = -9.1 points; |headless - body| = 1.9 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1440 | 1.000 | 0.000 | 0.182 |
| 0.1 | 0.4603 | 0.890 | 0.088 | 0.162 |
| 0.2 | 0.5062 | 0.766 | 0.162 | 0.162 |
| 0.3 | 0.5344 | 0.675 | 0.272 | 0.162 |
| 0.4 | 0.5525 | 0.545 | 0.338 | 0.130 |
| 0.5 | 0.5726 | 0.422 | 0.412 | 0.123 |
| 0.6 | 0.5912 | 0.318 | 0.507 | 0.117 |
| 0.7 | 0.6157 | 0.227 | 0.618 | 0.091 |
| 0.8 | 0.6469 | 0.156 | 0.750 | 0.078 |
| 0.9 | 0.6748 | 0.071 | 0.868 | 0.039 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.0586 | 1.000 | 0.000 | 0.292 |
| 0.1 | 0.0992 | 0.935 | 0.140 | 0.286 |
| 0.2 | 0.1176 | 0.857 | 0.265 | 0.266 |
| 0.3 | 0.1314 | 0.799 | 0.412 | 0.260 |
| 0.4 | 0.1480 | 0.701 | 0.515 | 0.240 |
| 0.5 | 0.1649 | 0.610 | 0.625 | 0.227 |
| 0.6 | 0.1816 | 0.513 | 0.728 | 0.201 |
| 0.7 | 0.2005 | 0.396 | 0.809 | 0.149 |
| 0.8 | 0.2284 | 0.266 | 0.875 | 0.110 |
| 0.9 | 0.2614 | 0.143 | 0.949 | 0.065 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1001 | 1.000 | 0.000 | 0.273 |
| 0.1 | 0.1658 | 0.922 | 0.125 | 0.260 |
| 0.2 | 0.2034 | 0.838 | 0.243 | 0.240 |
| 0.3 | 0.2178 | 0.760 | 0.368 | 0.214 |
| 0.4 | 0.2374 | 0.669 | 0.478 | 0.195 |
| 0.5 | 0.2570 | 0.571 | 0.581 | 0.162 |
| 0.6 | 0.2742 | 0.481 | 0.691 | 0.149 |
| 0.7 | 0.2920 | 0.338 | 0.743 | 0.097 |
| 0.8 | 0.3199 | 0.240 | 0.846 | 0.091 |
| 0.9 | 0.3501 | 0.117 | 0.919 | 0.052 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.0380 | 1.000 | 0.000 | 0.213 |
| 0.1 | 0.0720 | 0.852 | 0.082 | 0.213 |
| 0.2 | 0.0856 | 0.836 | 0.216 | 0.213 |
| 0.3 | 0.0942 | 0.689 | 0.299 | 0.197 |
| 0.4 | 0.1025 | 0.639 | 0.418 | 0.180 |
| 0.5 | 0.1108 | 0.557 | 0.522 | 0.148 |
| 0.6 | 0.1237 | 0.475 | 0.634 | 0.148 |
| 0.7 | 0.1381 | 0.393 | 0.739 | 0.148 |
| 0.8 | 0.1727 | 0.262 | 0.828 | 0.098 |
| 0.9 | 0.2095 | 0.115 | 0.903 | 0.016 |
