"""The experiment can answer the question: the ablation must DETECT where
identity lives, firing the correct kill-criterion verdict on corpora
constructed to have known ground truth."""

from __future__ import annotations

import run_ablation


def test_head_concentrated_corpus_triggers_kill(head_corpus):
    results = run_ablation.run_experiment(
        head_corpus, backbone="hist", arms=("head", "body", "headless"), seed=0
    )
    v = results["verdict"]
    assert v["verdict"] == run_ablation.VERDICT_KILL
    assert v["head_minus_headless"] >= 15.0


def test_distributed_corpus_triggers_distributed(distributed_corpus):
    results = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head", "body", "headless"), seed=0
    )
    v = results["verdict"]
    assert v["verdict"] == run_ablation.VERDICT_DISTRIBUTED
    assert v["headless_minus_body_abs"] <= 5.0
    assert v["head_minus_headless"] < 15.0


def test_experiment_deterministic(distributed_corpus):
    kwargs = dict(backbone="hist", arms=("head", "headless"), seed=7)
    a = run_ablation.run_experiment(distributed_corpus, **kwargs)
    b = run_ablation.run_experiment(distributed_corpus, **kwargs)
    for arm in kwargs["arms"]:
        for key in ("rank1", "rank5", "mAP", "open_set_auroc", "n_gallery", "n_known", "n_novel"):
            assert a["arms"][arm][key] == b["arms"][arm][key]


def test_verdict_requires_all_three_crop_arms(distributed_corpus):
    results = run_ablation.run_experiment(
        distributed_corpus, backbone="hist", arms=("head",), seed=0
    )
    assert results["verdict"]["verdict"] is None


def test_verdict_thresholds_exact_count_fractions():
    # 14/20 vs 11/20 is exactly 15.0 points but 100*(14/20 - 11/20) computes to
    # 14.999999999999993 in floats; the tolerance must keep KILL firing.
    def arms(head, headless, body):
        return {
            "head": {"rank1": head},
            "headless": {"rank1": headless},
            "body": {"rank1": body},
        }

    v = run_ablation.compute_verdict(arms(14 / 20, 11 / 20, 11 / 20))
    assert v["verdict"] == run_ablation.VERDICT_KILL

    # 11/20 vs 10/20 is exactly 5.0 points; distributed must still fire.
    v = run_ablation.compute_verdict(arms(12 / 20, 11 / 20, 10 / 20))
    assert v["verdict"] == run_ablation.VERDICT_DISTRIBUTED


def test_report_states_decision_rule(tmp_path):
    results = {
        "backbone": "hist",
        "seed": 0,
        "cutoff_fraction": 0.5,
        "arms": {
            "head": {"rank1": 0.9, "rank5": 0.9, "mAP": 0.9, "open_set_auroc": 0.9, "rejection_curve": [],
                     "n_gallery": 10, "n_known": 10, "n_novel": 5},
            "body": {"rank1": 0.5, "rank5": 0.6, "mAP": 0.5, "open_set_auroc": 0.8, "rejection_curve": [],
                     "n_gallery": 10, "n_known": 10, "n_novel": 5},
            "headless": {"rank1": 0.5, "rank5": 0.6, "mAP": 0.5, "open_set_auroc": 0.8, "rejection_curve": [],
                         "n_gallery": 10, "n_known": 10, "n_novel": 5},
        },
    }
    results["verdict"] = run_ablation.compute_verdict(results["arms"])
    results["caveat"] = run_ablation.CAVEAT
    run_ablation.write_report(results, str(tmp_path))
    text = (tmp_path / "report.md").read_text()
    assert ">= 15 points -> KILL/redirect" in text
    assert "|headless - body| <= 5 points" in text


def test_verdict_inconclusive_below_decision_floor():
    # The real 2026-08-31 Melops zero-shot run: all crop arms at 1-2 points.
    def arms(head, headless, body):
        return {
            "head": {"rank1": head},
            "headless": {"rank1": headless},
            "body": {"rank1": body},
        }

    v = run_ablation.compute_verdict(arms(0.019, 0.012, 0.010))
    assert v["verdict"] == run_ablation.VERDICT_INCONCLUSIVE

    # At a healthy operating point the original rules still apply.
    v = run_ablation.compute_verdict(arms(0.40, 0.20, 0.22))
    assert v["verdict"] == run_ablation.VERDICT_KILL
    v = run_ablation.compute_verdict(arms(0.40, 0.36, 0.38))
    assert v["verdict"] == run_ablation.VERDICT_DISTRIBUTED

    # Exactly at the floor with one arm at 15.0 points: rules express again.
    v = run_ablation.compute_verdict(arms(15 / 100, 0.0, 0.0))
    assert v["verdict"] == run_ablation.VERDICT_KILL
