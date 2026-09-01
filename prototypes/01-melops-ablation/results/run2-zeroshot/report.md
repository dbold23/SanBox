# Melops Phase 1A ablation report

Backbone: `megadescriptor` | seed 0 | cutoff fraction 0.50

| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |
|---|---|---|---|---|---|---|---|
| cross_orientation | 5168 | 1281 | 5962 | 0.007 | 0.022 | 0.018 | 0.391 |

## Kill-criterion verdict

Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by
>= 15 points -> KILL/redirect (identity concentrated in the rigid part;
build a patch matcher, not a surface). |headless - body| <= 5 points ->
identity distributed (Approach 2 earns a hearing). Otherwise ->
intermediate (widen data before deciding). KILL is checked first.
A decision floor gates all of it: if every crop arm is below 15
Rank-1 points the verdict is INCONCLUSIVE, because the kill rule
cannot express and the distributed rule is vacuous at that level.

No verdict: arms missing or without known queries: ['head', 'body', 'headless']

## Caveat

Do not overread: Melops fish are board-mounted (photographed against a standardised white board) and are not laterally bending in frame. This experiment settles WHERE identity lives (rigid part vs deformable flank), never what unwrapping buys on a bending body.

## Rejection curves (Rank-1 at max-similarity threshold quantiles)

### cross_orientation

| quantile | threshold | known accept | novel reject | Rank-1@thr |
|---|---|---|---|---|
| 0.0 | 0.2933 | 1.000 | 0.000 | 0.007 |
| 0.1 | 0.6317 | 0.867 | 0.093 | 0.007 |
| 0.2 | 0.6745 | 0.707 | 0.180 | 0.006 |
| 0.3 | 0.7046 | 0.571 | 0.272 | 0.005 |
| 0.4 | 0.7271 | 0.445 | 0.367 | 0.004 |
| 0.5 | 0.7454 | 0.351 | 0.468 | 0.004 |
| 0.6 | 0.7614 | 0.262 | 0.570 | 0.003 |
| 0.7 | 0.7775 | 0.196 | 0.678 | 0.003 |
| 0.8 | 0.7923 | 0.132 | 0.785 | 0.002 |
| 0.9 | 0.8103 | 0.066 | 0.893 | 0.002 |
