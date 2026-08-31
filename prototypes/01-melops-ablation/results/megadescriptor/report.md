# Melops Phase 1A ablation report

Backbone: `megadescriptor` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 10410 | 2596 | 11276 | 0.019 | 0.052 | 0.040 | 0.336 |
| body | 10410 | 2596 | 11276 | 0.010 | 0.026 | 0.022 | 0.354 |
| headless | 10410 | 2596 | 11276 | 0.012 | 0.039 | 0.029 | 0.407 |
| cross_orientation | 5168 | 6503 | 5962 | 0.186 | 0.325 | 0.256 | 0.555 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.

**identity distributed - Approach 2 earns a hearing**

head - headless = 0.8 points; |headless - body| = 0.2 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2830 | 1.000 | 0.000 | 0.019 |
| 0.1 | 0.6679 | 0.819 | 0.081 | 0.017 |
| 0.2 | 0.7022 | 0.654 | 0.166 | 0.013 |
| 0.3 | 0.7257 | 0.520 | 0.259 | 0.010 |
| 0.4 | 0.7455 | 0.409 | 0.356 | 0.009 |
| 0.5 | 0.7618 | 0.306 | 0.455 | 0.007 |
| 0.6 | 0.7776 | 0.216 | 0.558 | 0.005 |
| 0.7 | 0.7921 | 0.137 | 0.662 | 0.004 |
| 0.8 | 0.8072 | 0.080 | 0.772 | 0.002 |
| 0.9 | 0.8251 | 0.033 | 0.885 | 0.002 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.3830 | 1.000 | 0.000 | 0.010 |
| 0.1 | 0.6478 | 0.817 | 0.081 | 0.008 |
| 0.2 | 0.6884 | 0.666 | 0.169 | 0.008 |
| 0.3 | 0.7159 | 0.537 | 0.262 | 0.007 |
| 0.4 | 0.7375 | 0.421 | 0.359 | 0.006 |
| 0.5 | 0.7562 | 0.324 | 0.460 | 0.005 |
| 0.6 | 0.7731 | 0.238 | 0.563 | 0.005 |
| 0.7 | 0.7899 | 0.169 | 0.670 | 0.004 |
| 0.8 | 0.8055 | 0.105 | 0.778 | 0.002 |
| 0.9 | 0.8242 | 0.046 | 0.887 | 0.002 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2823 | 1.000 | 0.000 | 0.012 |
| 0.1 | 0.6281 | 0.843 | 0.087 | 0.010 |
| 0.2 | 0.6698 | 0.707 | 0.179 | 0.008 |
| 0.3 | 0.6961 | 0.581 | 0.273 | 0.008 |
| 0.4 | 0.7173 | 0.476 | 0.371 | 0.008 |
| 0.5 | 0.7343 | 0.388 | 0.474 | 0.007 |
| 0.6 | 0.7507 | 0.299 | 0.577 | 0.007 |
| 0.7 | 0.7661 | 0.225 | 0.683 | 0.005 |
| 0.8 | 0.7822 | 0.146 | 0.788 | 0.004 |
| 0.9 | 0.8012 | 0.080 | 0.895 | 0.002 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2933 | 1.000 | 0.000 | 0.186 |
| 0.1 | 0.6471 | 0.913 | 0.114 | 0.177 |
| 0.2 | 0.6879 | 0.817 | 0.218 | 0.167 |
| 0.3 | 0.7156 | 0.718 | 0.320 | 0.154 |
| 0.4 | 0.7375 | 0.624 | 0.427 | 0.138 |
| 0.5 | 0.7553 | 0.529 | 0.532 | 0.122 |
| 0.6 | 0.7717 | 0.437 | 0.640 | 0.107 |
| 0.7 | 0.7865 | 0.343 | 0.747 | 0.088 |
| 0.8 | 0.8013 | 0.241 | 0.845 | 0.070 |
| 0.9 | 0.8193 | 0.134 | 0.937 | 0.048 |
