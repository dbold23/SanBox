# Melops run-2 diagnostics

Backbone: `megadescriptor` | analysis arm: `body` | seed 0 | cutoff fraction 0.50

Gallery 10410 units | 2596 known / 11276 novel queries | 295 same-date near-duplicates excluded.

The recapture-gap curve below doubles as a PATTERN-STABILITY measurement:
it tracks how the true-mate similarity of the same (identity, side) unit
decays with elapsed time, which is independently valuable beyond debugging
run 1's inconclusive numbers.

## a. Recapture-gap curve (known queries vs their gallery image)

| gap (days) | n | mean true-mate cosine sim | Rank-1 |
|---|---|---|---|
| 0-30 | 213 | 0.605 | 0.052 |
| 31-180 | 513 | 0.531 | 0.008 |
| 181-365 | 611 | 0.500 | 0.008 |
| 366-730 | 688 | 0.477 | 0.006 |
| 731+ | 571 | 0.474 | 0.004 |

## b. Open-set AUROC strata by query year

Tests the ANALYSIS.md temporal-confound hypothesis (known queries span
all years; novel queries are late-years only, so acquisition drift can
invert the pooled AUROC even when every within-year comparison is sane).

| query year | n_known | n_novel | AUROC | mean max-sim (known) | mean max-sim (novel) |
|---|---|---|---|---|---|
| 2018 | 212 | 0 | n/a | 0.760 | n/a |
| 2019 | 101 | 0 | n/a | 0.744 | n/a |
| 2020 | 838 | 0 | n/a | 0.731 | n/a |
| 2021 | 760 | 1565 | 0.371 | 0.689 | 0.723 |
| 2022 | 378 | 3229 | 0.279 | 0.715 | 0.764 |
| 2023 | 217 | 4420 | 0.219 | 0.690 | 0.760 |
| 2024 | 90 | 2062 | 0.468 | 0.723 | 0.734 |

Pooled AUROC: 0.354

**Automated reading: below 0.5 even within years => confound NOT temporal, investigate further [3 of 7 strata untestable (single-class) and excluded]**

## c. Small-gallery calibration (NLDL one-shot 0.35 anchor comparability)

K = 500 enrolled + 500 novel units (requested 500), 3 subsample seeds.

| metric | mean | min | max |
|---|---|---|---|
| Rank-1 | 0.053 | 0.034 | 0.078 |
| open-set AUROC | 0.370 | 0.343 | 0.400 |

compare against the NLDL 2023 one-shot 0.35 anchor (arXiv:2301.00596, population-trained model, small gallery) [SEARCH-grade, confirm against the PDF before quoting]

## d. Crop contact sheets

* `head`: `contact_sheet_head.png` -- eyeball that head crops contain what they claim.
* `body`: `contact_sheet_body.png` -- eyeball that body crops contain what they claim.
* `headless`: `contact_sheet_headless.png` -- eyeball that headless crops contain what they claim.
