# Melops Phase 1A ablation report

Backbone: `dinov2` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 10410 | 2596 | 11276 | 0.011 | 0.034 | 0.027 | 0.307 |
| body | 10410 | 2596 | 11276 | 0.012 | 0.032 | 0.026 | 0.318 |
| headless | 10410 | 2596 | 11276 | 0.009 | 0.025 | 0.022 | 0.352 |
| cross_orientation | 5168 | 6503 | 5962 | 0.200 | 0.344 | 0.272 | 0.499 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.

**identity distributed - Approach 2 earns a hearing**

head - headless = 0.2 points; |headless - body| = 0.3 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.6232 | 1.000 | 0.000 | 0.011 |
| 0.1 | 0.9158 | 0.817 | 0.081 | 0.010 |
| 0.2 | 0.9252 | 0.630 | 0.161 | 0.007 |
| 0.3 | 0.9311 | 0.476 | 0.248 | 0.005 |
| 0.4 | 0.9357 | 0.360 | 0.345 | 0.005 |
| 0.5 | 0.9396 | 0.260 | 0.445 | 0.004 |
| 0.6 | 0.9430 | 0.183 | 0.550 | 0.003 |
| 0.7 | 0.9464 | 0.113 | 0.657 | 0.001 |
| 0.8 | 0.9501 | 0.069 | 0.770 | 0.001 |
| 0.9 | 0.9542 | 0.033 | 0.885 | 0.000 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.5636 | 1.000 | 0.000 | 0.012 |
| 0.1 | 0.9236 | 0.816 | 0.081 | 0.010 |
| 0.2 | 0.9336 | 0.656 | 0.167 | 0.008 |
| 0.3 | 0.9394 | 0.512 | 0.257 | 0.007 |
| 0.4 | 0.9438 | 0.380 | 0.349 | 0.005 |
| 0.5 | 0.9476 | 0.275 | 0.448 | 0.005 |
| 0.6 | 0.9509 | 0.187 | 0.551 | 0.003 |
| 0.7 | 0.9541 | 0.123 | 0.659 | 0.003 |
| 0.8 | 0.9576 | 0.062 | 0.768 | 0.001 |
| 0.9 | 0.9614 | 0.024 | 0.882 | 0.001 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.6247 | 1.000 | 0.000 | 0.009 |
| 0.1 | 0.9159 | 0.839 | 0.086 | 0.008 |
| 0.2 | 0.9274 | 0.690 | 0.175 | 0.007 |
| 0.3 | 0.9335 | 0.553 | 0.266 | 0.007 |
| 0.4 | 0.9383 | 0.426 | 0.360 | 0.005 |
| 0.5 | 0.9421 | 0.316 | 0.458 | 0.004 |
| 0.6 | 0.9456 | 0.223 | 0.559 | 0.003 |
| 0.7 | 0.9490 | 0.143 | 0.664 | 0.003 |
| 0.8 | 0.9524 | 0.084 | 0.773 | 0.002 |
| 0.9 | 0.9564 | 0.039 | 0.886 | 0.001 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.6056 | 1.000 | 0.000 | 0.200 |
| 0.1 | 0.8801 | 0.912 | 0.113 | 0.192 |
| 0.2 | 0.8924 | 0.806 | 0.207 | 0.179 |
| 0.3 | 0.8994 | 0.702 | 0.302 | 0.163 |
| 0.4 | 0.9048 | 0.594 | 0.394 | 0.145 |
| 0.5 | 0.9090 | 0.493 | 0.492 | 0.127 |
| 0.6 | 0.9127 | 0.392 | 0.591 | 0.105 |
| 0.7 | 0.9163 | 0.294 | 0.694 | 0.082 |
| 0.8 | 0.9202 | 0.195 | 0.795 | 0.058 |
| 0.9 | 0.9249 | 0.097 | 0.897 | 0.033 |
