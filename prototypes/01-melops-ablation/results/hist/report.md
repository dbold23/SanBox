# Melops Phase 1A ablation report

Backbone: `hist` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 10410 | 2596 | 11276 | 0.003 | 0.009 | 0.008 | 0.352 |
| body | 10410 | 2596 | 11276 | 0.001 | 0.004 | 0.004 | 0.323 |
| headless | 10410 | 2596 | 11276 | 0.000 | 0.003 | 0.004 | 0.349 |
| cross_orientation | 5168 | 6503 | 5962 | 0.002 | 0.005 | 0.006 | 0.527 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.

**identity distributed - Approach 2 earns a hearing**

head - headless = 0.2 points; |headless - body| = 0.0 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.6059 | 1.000 | 0.000 | 0.003 |
| 0.1 | 0.9684 | 0.823 | 0.082 | 0.003 |
| 0.2 | 0.9740 | 0.664 | 0.169 | 0.002 |
| 0.3 | 0.9771 | 0.523 | 0.259 | 0.001 |
| 0.4 | 0.9793 | 0.416 | 0.358 | 0.001 |
| 0.5 | 0.9811 | 0.318 | 0.458 | 0.000 |
| 0.6 | 0.9827 | 0.235 | 0.562 | 0.000 |
| 0.7 | 0.9842 | 0.167 | 0.669 | 0.000 |
| 0.8 | 0.9857 | 0.111 | 0.779 | 0.000 |
| 0.9 | 0.9874 | 0.047 | 0.888 | 0.000 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.6867 | 1.000 | 0.000 | 0.001 |
| 0.1 | 0.9311 | 0.833 | 0.085 | 0.000 |
| 0.2 | 0.9445 | 0.675 | 0.171 | 0.000 |
| 0.3 | 0.9526 | 0.519 | 0.258 | 0.000 |
| 0.4 | 0.9579 | 0.388 | 0.351 | 0.000 |
| 0.5 | 0.9620 | 0.275 | 0.448 | 0.000 |
| 0.6 | 0.9657 | 0.189 | 0.551 | 0.000 |
| 0.7 | 0.9690 | 0.116 | 0.658 | 0.000 |
| 0.8 | 0.9722 | 0.063 | 0.768 | 0.000 |
| 0.9 | 0.9758 | 0.026 | 0.883 | 0.000 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.5664 | 1.000 | 0.000 | 0.000 |
| 0.1 | 0.9226 | 0.838 | 0.086 | 0.000 |
| 0.2 | 0.9405 | 0.691 | 0.175 | 0.000 |
| 0.3 | 0.9493 | 0.546 | 0.265 | 0.000 |
| 0.4 | 0.9553 | 0.422 | 0.359 | 0.000 |
| 0.5 | 0.9599 | 0.312 | 0.457 | 0.000 |
| 0.6 | 0.9640 | 0.216 | 0.558 | 0.000 |
| 0.7 | 0.9675 | 0.141 | 0.663 | 0.000 |
| 0.8 | 0.9708 | 0.084 | 0.773 | 0.000 |
| 0.9 | 0.9744 | 0.030 | 0.884 | 0.000 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1446 | 1.000 | 0.000 | 0.002 |
| 0.1 | 0.8035 | 0.924 | 0.126 | 0.002 |
| 0.2 | 0.8183 | 0.821 | 0.223 | 0.002 |
| 0.3 | 0.8292 | 0.721 | 0.323 | 0.002 |
| 0.4 | 0.8380 | 0.618 | 0.419 | 0.002 |
| 0.5 | 0.8460 | 0.516 | 0.518 | 0.002 |
| 0.6 | 0.8536 | 0.411 | 0.612 | 0.002 |
| 0.7 | 0.8612 | 0.308 | 0.709 | 0.002 |
| 0.8 | 0.8694 | 0.203 | 0.803 | 0.001 |
| 0.9 | 0.8799 | 0.103 | 0.904 | 0.001 |
