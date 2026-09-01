# Melops Phase 1A ablation report

Backbone: `finetuned:/root/runs/ft-mega/checkpoint.pt` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| head | 4182 | 1055 | 4480 | 0.040 | 0.078 | 0.066 | 0.351 |
| body | 4182 | 1055 | 4480 | 0.019 | 0.069 | 0.051 | 0.416 |
| headless | 4182 | 1055 | 4480 | 0.023 | 0.079 | 0.055 | 0.492 |
| cross_orientation | 2070 | 520 | 2352 | 0.012 | 0.037 | 0.035 | 0.459 |

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

head - headless = 1.7 points; |headless - body| = 0.4 points.

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### head

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2604 | 1.000 | 0.000 | 0.040 |
| 0.1 | 0.6312 | 0.829 | 0.083 | 0.035 |
| 0.2 | 0.6686 | 0.680 | 0.172 | 0.028 |
| 0.3 | 0.6961 | 0.544 | 0.263 | 0.025 |
| 0.4 | 0.7181 | 0.430 | 0.360 | 0.020 |
| 0.5 | 0.7375 | 0.308 | 0.455 | 0.017 |
| 0.6 | 0.7553 | 0.227 | 0.559 | 0.014 |
| 0.7 | 0.7721 | 0.150 | 0.665 | 0.009 |
| 0.8 | 0.7883 | 0.087 | 0.773 | 0.008 |
| 0.9 | 0.8060 | 0.038 | 0.885 | 0.006 |

### body

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2354 | 1.000 | 0.000 | 0.019 |
| 0.1 | 0.4285 | 0.839 | 0.086 | 0.018 |
| 0.2 | 0.4645 | 0.700 | 0.177 | 0.016 |
| 0.3 | 0.4927 | 0.585 | 0.273 | 0.016 |
| 0.4 | 0.5152 | 0.494 | 0.375 | 0.014 |
| 0.5 | 0.5362 | 0.395 | 0.475 | 0.012 |
| 0.6 | 0.5584 | 0.319 | 0.581 | 0.009 |
| 0.7 | 0.5808 | 0.245 | 0.687 | 0.009 |
| 0.8 | 0.6065 | 0.154 | 0.789 | 0.004 |
| 0.9 | 0.6405 | 0.082 | 0.896 | 0.002 |

### headless

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1917 | 1.000 | 0.000 | 0.023 |
| 0.1 | 0.4639 | 0.889 | 0.098 | 0.021 |
| 0.2 | 0.5007 | 0.773 | 0.194 | 0.019 |
| 0.3 | 0.5262 | 0.657 | 0.290 | 0.018 |
| 0.4 | 0.5490 | 0.566 | 0.392 | 0.018 |
| 0.5 | 0.5685 | 0.490 | 0.498 | 0.016 |
| 0.6 | 0.5884 | 0.401 | 0.600 | 0.015 |
| 0.7 | 0.6084 | 0.316 | 0.704 | 0.012 |
| 0.8 | 0.6326 | 0.211 | 0.803 | 0.009 |
| 0.9 | 0.6638 | 0.123 | 0.905 | 0.005 |

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.1292 | 1.000 | 0.000 | 0.012 |
| 0.1 | 0.4064 | 0.875 | 0.095 | 0.010 |
| 0.2 | 0.4454 | 0.762 | 0.192 | 0.010 |
| 0.3 | 0.4713 | 0.642 | 0.287 | 0.010 |
| 0.4 | 0.4933 | 0.523 | 0.383 | 0.010 |
| 0.5 | 0.5121 | 0.423 | 0.483 | 0.008 |
| 0.6 | 0.5323 | 0.344 | 0.588 | 0.008 |
| 0.7 | 0.5555 | 0.265 | 0.692 | 0.004 |
| 0.8 | 0.5799 | 0.192 | 0.798 | 0.004 |
| 0.9 | 0.6119 | 0.110 | 0.902 | 0.004 |
