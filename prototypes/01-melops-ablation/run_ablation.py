"""CLI for the Phase 1A Melops rigid-part vs deformable-body ablation.

Usage:
    python run_ablation.py --data synthetic --root corpus/ --backbone hist \\
        --arms head,body,headless,cross_orientation --out results/

Arms ``head`` / ``body`` / ``headless`` share one identical split (the split
depends only on identity/side/date, never on the crop) so their Rank-1 numbers
are directly comparable. The ``cross_orientation`` arm enrolls one side and
queries the other; it is THE ONLY arm that ignores the same-side matching
rule, by design.

Outputs ``results.json`` and ``report.md`` in ``--out``, including the
kill-criterion verdict from the Phase 1A spec.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import embedders
import melops_data
import protocol

CROP_ARMS = ("head", "body", "headless")
ALL_ARMS = CROP_ARMS + ("cross_orientation",)

VERDICT_KILL = (
    "KILL / redirect: identity concentrated in rigid part - build a patch "
    "matcher, not a surface"
)
VERDICT_DISTRIBUTED = "identity distributed - Approach 2 earns a hearing"
VERDICT_INTERMEDIATE = "intermediate - widen data before deciding"
VERDICT_INCONCLUSIVE = (
    "INCONCLUSIVE - operating point below decision floor: every crop arm is "
    "under 15 Rank-1 points, so the >= 15-point kill criterion is arithmetically "
    "inexpressible and the <= 5-point distributed rule is vacuous on a "
    "near-floor base. Improve the matcher (fine-tune on a disjoint identity "
    "subset) before reading the ablation; do not cite these deltas either way."
)

CAVEAT = (
    "Do not overread: Melops fish are board-mounted (photographed against a "
    "standardised white board) and are not laterally bending in frame. This "
    "experiment settles WHERE identity lives (rigid part vs deformable "
    "flank), never what unwrapping buys on a bending body."
)


def _embed_frame(embedder, root, frame):
    crops = [melops_data.load_crop(root, row) for _, row in frame.iterrows()]
    return embedder.embed(crops)


def run_experiment(root, backbone="hist", arms=ALL_ARMS, seed=0, cutoff_fraction=0.5):
    """Run the requested arms; returns the full results dict (JSON-safe)."""
    for arm in arms:
        if arm not in ALL_ARMS:
            raise ValueError("unknown arm %r; choose from %r" % (arm, ALL_ARMS))
    embedder = embedders.get_embedder(backbone, seed=seed)
    results = {
        "backbone": backbone,
        "seed": int(seed),
        "cutoff_fraction": float(cutoff_fraction),
        "root": os.path.abspath(root),
        "arms": {},
    }
    for arm in arms:
        t0 = time.time()
        if arm == "cross_orientation":
            df = melops_data.load_melops(root, bbox="body")
            gallery_df, query_df = protocol.cross_orientation_split(
                df, enroll_side="L", query_side="R", cutoff_fraction=cutoff_fraction, seed=seed
            )
            cross = True
        else:
            df = melops_data.load_melops(root, bbox=arm)
            gallery_df, query_df = protocol.one_shot_open_set_split(
                df, cutoff_fraction=cutoff_fraction, seed=seed
            )
            cross = False
        gallery_emb = _embed_frame(embedder, root, gallery_df)
        query_emb = _embed_frame(embedder, root, query_df)
        metrics = protocol.evaluate(gallery_emb, gallery_df, query_emb, query_df, cross_side=cross)
        metrics["n_same_date_excluded"] = int(query_df.attrs.get("n_same_date_excluded", 0))
        metrics["elapsed_s"] = round(time.time() - t0, 2)
        results["arms"][arm] = metrics
    results["verdict"] = compute_verdict(results["arms"])
    results["caveat"] = CAVEAT
    return results


def compute_verdict(arm_results):
    """Phase 1A kill criterion, in Rank-1 percentage points.

    Requires the head, body and headless arms; otherwise returns None with a
    note. Order matters: the >= 15-point head-vs-headless kill fires first.
    """
    needed = [a for a in CROP_ARMS if a not in arm_results or arm_results[a].get("rank1") is None]
    if needed:
        return {"verdict": None, "note": "arms missing or without known queries: %r" % needed}
    head = 100.0 * arm_results["head"]["rank1"]
    body = 100.0 * arm_results["body"]["rank1"]
    headless = 100.0 * arm_results["headless"]["rank1"]
    # Rank-1 is a count ratio, so 100*rank1 carries float error of ~1e-13 that
    # can flip an exactly-at-threshold comparison (e.g. 14/20 vs 11/20). The
    # tolerance is far above that error and far below the 5-point grid of any
    # realistic count difference.
    tol = 1e-9
    # Decision floor: the kill rule needs head >= 15 points to be expressible
    # at all, and "within 5 points" on a ~1% base is noise, not evidence of
    # distributed identity. Below the floor the experiment has not answered
    # the question -- say so instead of laundering a floor artifact into a
    # programme decision. (The real Melops zero-shot run of 2026-08-31 sat at
    # 1-2 points across all crop arms and is exactly this case.)
    if max(head, body, headless) < 15.0 - tol:
        verdict = VERDICT_INCONCLUSIVE
    elif head - headless >= 15.0 - tol:
        verdict = VERDICT_KILL
    elif abs(headless - body) <= 5.0 + tol:
        verdict = VERDICT_DISTRIBUTED
    else:
        verdict = VERDICT_INTERMEDIATE
    return {
        "verdict": verdict,
        "rank1_points": {"head": head, "body": body, "headless": headless},
        "head_minus_headless": head - headless,
        "headless_minus_body_abs": abs(headless - body),
    }


def _fmt(value):
    return "n/a" if value is None else "%.3f" % value


def write_report(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Melops Phase 1A ablation report",
        "",
        "Backbone: `%s` | seed %d | cutoff fraction %.2f" % (
            results["backbone"], results["seed"], results["cutoff_fraction"]),
        "",
        "| arm | n_gallery | n_known | n_novel | Rank-1 | Rank-5 | mAP | open-set AUROC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, m in results["arms"].items():
        lines.append(
            "| %s | %d | %d | %d | %s | %s | %s | %s |"
            % (arm, m["n_gallery"], m["n_known"], m["n_novel"],
               _fmt(m["rank1"]), _fmt(m["rank5"]), _fmt(m["mAP"]), _fmt(m["open_set_auroc"]))
        )
    lines += ["", "## Kill-criterion verdict", ""]
    lines += [
        "Decision rule (Phase 1A spec): head Rank-1 exceeding headless Rank-1 by",
        ">= 15 points -> KILL/redirect (identity concentrated in the rigid part;",
        "build a patch matcher, not a surface). |headless - body| <= 5 points ->",
        "identity distributed (Approach 2 earns a hearing). Otherwise ->",
        "intermediate (widen data before deciding). KILL is checked first.",
        "A decision floor gates all of it: if every crop arm is below 15",
        "Rank-1 points the verdict is INCONCLUSIVE, because the kill rule",
        "cannot express and the distributed rule is vacuous at that level.",
        "",
    ]
    v = results["verdict"]
    if v["verdict"] is None:
        lines.append("No verdict: %s" % v["note"])
    else:
        lines.append("**%s**" % v["verdict"])
        lines.append("")
        lines.append(
            "head - headless = %.1f points; |headless - body| = %.1f points."
            % (v["head_minus_headless"], v["headless_minus_body_abs"])
        )
    lines += ["", "## Caveat", "", CAVEAT, ""]
    lines += [
        "## Rejection curves (Rank-1 at max-similarity threshold quantiles)",
        "",
    ]
    for arm, m in results["arms"].items():
        lines.append("### %s" % arm)
        lines.append("")
        lines.append("| quantile | threshold | known accept | novel reject | Rank-1@thr |")
        lines.append("|---|---|---|---|---|")
        for row in m["rejection_curve"]:
            lines.append(
                "| %.1f | %.4f | %s | %s | %s |"
                % (row["quantile"], row["threshold"],
                   _fmt(row["known_acceptance_rate"]), _fmt(row["novel_rejection_rate"]),
                   _fmt(row["rank1_at_threshold"]))
            )
        lines.append("")
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return json_path, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("synthetic", "melops"), required=True)
    parser.add_argument("--root", required=True, help="corpus root directory")
    parser.add_argument("--backbone", default="hist",
                        choices=("hist", "random", "megadescriptor", "dinov2", "miewid"))
    parser.add_argument("--arms", default="head,body,headless,cross_orientation")
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff-fraction", type=float, default=0.5)
    parser.add_argument("--n-individuals", type=int, default=40,
                        help="synthetic only; used when --root has no metadata.csv yet")
    parser.add_argument("--head-signal", type=float, default=1.0, help="synthetic only")
    parser.add_argument("--body-signal", type=float, default=1.0, help="synthetic only")
    args = parser.parse_args(argv)

    if args.data == "synthetic" and not os.path.exists(os.path.join(args.root, "metadata.csv")):
        melops_data.make_synthetic(
            args.root,
            n_individuals=args.n_individuals,
            seed=args.seed,
            head_signal=args.head_signal,
            body_signal=args.body_signal,
        )

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    results = run_experiment(
        args.root, backbone=args.backbone, arms=arms,
        seed=args.seed, cutoff_fraction=args.cutoff_fraction,
    )
    json_path, report_path = write_report(results, args.out)

    for arm, m in results["arms"].items():
        print("%-18s Rank-1=%s Rank-5=%s mAP=%s AUROC=%s (gallery=%d known=%d novel=%d)"
              % (arm, _fmt(m["rank1"]), _fmt(m["rank5"]), _fmt(m["mAP"]),
                 _fmt(m["open_set_auroc"]), m["n_gallery"], m["n_known"], m["n_novel"]))
    v = results["verdict"]
    print("VERDICT: %s" % (v["verdict"] if v["verdict"] is not None else v["note"]))
    print("CAVEAT: %s" % CAVEAT)
    print("Wrote %s and %s" % (json_path, report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
