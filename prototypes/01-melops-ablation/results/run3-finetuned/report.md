# Melops Phase 1A ablation report

Backbone: `finetuned:/root/runs/ft-mega3/checkpoint.pt` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 4182 | 1055 | 4480 | 0.048 | 0.098 | 0.081 | 0.365 |
| body | 4182 | 1055 | 4480 | 0.072 | 0.155 | 0.121 | 0.567 |
| headless | 4182 | 1055 | 4480 | 0.072 | 0.140 | 0.115 | 0.496 |
| cross_orientation | 2070 | 520 | 2352 | 0.044 | 0.115 | 0.085 | 0.627 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.
A decision floor gates all of it: if every crop arm is below 15
Rank-1 points the verdict is INCONCLUSIVE, because the kill rule
cannot express and the distributed rule is vacuous at that level.

**INCONCLUSIVE - operating point below decision floor: every crop arm is under 15 Rank-1 points, so the >= 15-point kill criterion is arithmetically inexpressible and the <= 5-point distributed rule is vacuous on a near-floor base. Improve the matcher (fine-tune on a disjoint identity subset) before reading the ablation; do not cite these deltas either way.**

head - headless = -2.4 points; |headless - body| = 0.0 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2781 | 1.000 | 0.000 | 0.048 |
| 0.1 | 0.6080 | 0.826 | 0.083 | 0.046 |
| 0.2 | 0.6440 | 0.680 | 0.172 | 0.042 |
| 0.3 | 0.6691 | 0.557 | 0.267 | 0.037 |
| 0.4 | 0.6904 | 0.452 | 0.365 | 0.032 |
| 0.5 | 0.7096 | 0.343 | 0.463 | 0.027 |
| 0.6 | 0.7263 | 0.254 | 0.566 | 0.023 |
| 0.7 | 0.7438 | 0.175 | 0.671 | 0.019 |
| 0.8 | 0.7596 | 0.097 | 0.776 | 0.011 |
| 0.9 | 0.7783 | 0.051 | 0.888 | 0.009 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1615 | 1.000 | 0.000 | 0.072 |
| 0.1 | 0.2626 | 0.902 | 0.101 | 0.070 |
| 0.2 | 0.2852 | 0.821 | 0.205 | 0.064 |
| 0.3 | 0.3016 | 0.736 | 0.309 | 0.064 |
| 0.4 | 0.3162 | 0.646 | 0.411 | 0.060 |
| 0.5 | 0.3299 | 0.562 | 0.515 | 0.057 |
| 0.6 | 0.3444 | 0.486 | 0.620 | 0.053 |
| 0.7 | 0.3598 | 0.396 | 0.723 | 0.050 |
| 0.8 | 0.3804 | 0.302 | 0.824 | 0.045 |
| 0.9 | 0.4106 | 0.164 | 0.915 | 0.027 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2087 | 1.000 | 0.000 | 0.072 |
| 0.1 | 0.3352 | 0.896 | 0.099 | 0.071 |
| 0.2 | 0.3629 | 0.777 | 0.195 | 0.070 |
| 0.3 | 0.3820 | 0.669 | 0.293 | 0.066 |
| 0.4 | 0.3995 | 0.567 | 0.392 | 0.064 |
| 0.5 | 0.4145 | 0.478 | 0.495 | 0.060 |
| 0.6 | 0.4309 | 0.405 | 0.601 | 0.054 |
| 0.7 | 0.4476 | 0.312 | 0.703 | 0.043 |
| 0.8 | 0.4673 | 0.209 | 0.802 | 0.032 |
| 0.9 | 0.4957 | 0.125 | 0.906 | 0.023 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1173 | 1.000 | 0.000 | 0.044 |
| 0.1 | 0.1888 | 0.954 | 0.112 | 0.044 |
| 0.2 | 0.2081 | 0.881 | 0.218 | 0.042 |
| 0.3 | 0.2250 | 0.827 | 0.328 | 0.040 |
| 0.4 | 0.2392 | 0.744 | 0.432 | 0.033 |
| 0.5 | 0.2536 | 0.654 | 0.534 | 0.029 |
| 0.6 | 0.2679 | 0.550 | 0.633 | 0.027 |
| 0.7 | 0.2828 | 0.431 | 0.729 | 0.023 |
| 0.8 | 0.3012 | 0.317 | 0.826 | 0.019 |
| 0.9 | 0.3292 | 0.167 | 0.915 | 0.012 |
