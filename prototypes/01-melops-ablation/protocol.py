"""One-shot open-set evaluation protocol for the Melops ablation.

Identity unit
-------------
The enrollment unit is ``(identity, side)``. Melops tracks ``side`` as a
first-class column (``orientation == side``) and the spec requires both flanks
to be tracked as separate identities; matching is only ever within the same
side value. The single deliberate exception is the cross-orientation arm
(``cross_orientation_split`` + ``evaluate(..., cross_side=True)``), which
enrolls one side and queries the other BY DESIGN.

Invariants (raised on violation, not silently patched):
* no image is ever both gallery and query;
* a known query's ``(identity, side)`` is in the gallery, a novel query's is
  not (open set);
* every similarity is computed within one side (unless ``cross_side=True``);
* the gallery holds exactly one image per enrolled ``(identity, side)``
  (one-shot).

All randomness is confined to explicit seeds (used only to break exact
date ties deterministically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ProtocolViolation(RuntimeError):
    """A split or evaluation invariant was violated."""


def _check_input_frame(df):
    required = ("image_id", "identity", "date", "side")
    for col in required:
        if col not in df.columns:
            raise ProtocolViolation("input frame missing column %r" % col)
    if df["image_id"].duplicated().any():
        dup = df.loc[df["image_id"].duplicated(), "image_id"].iloc[0]
        raise ProtocolViolation("duplicate image_id %r in input frame" % dup)
    # one image must map to one identity: image_id is unique, but also guard
    # against the same underlying path listed under two identities
    if "path" in df.columns and df["path"].duplicated().any():
        dup = df.loc[df["path"].duplicated(), "path"].iloc[0]
        raise ProtocolViolation("same image path %r listed more than once" % dup)


def _resolve_cutoff(dates, cutoff_fraction, cutoff_date):
    if (cutoff_fraction is None) == (cutoff_date is None):
        raise ValueError("pass exactly one of cutoff_fraction, cutoff_date")
    if cutoff_date is not None:
        return pd.Timestamp(cutoff_date)
    if not (0.0 < float(cutoff_fraction) < 1.0):
        raise ValueError("cutoff_fraction must be in (0, 1)")
    return dates.quantile(float(cutoff_fraction))


def one_shot_open_set_split(df, cutoff_fraction=None, cutoff_date=None, seed=0,
                            same_date_policy="exclude"):
    """Time-separated one-shot open-set split over (identity, side) units.

    Gallery: for each (identity, side) whose earliest sighting on that side is
    strictly before the cutoff, its single earliest image (exact-date ties
    broken deterministically from ``seed``). One image per unit -- one-shot.

    Queries: every other image. A query is *known* if its (identity, side) is
    enrolled, *novel* if its unit's first sighting is on/after the cutoff
    (open set). Singletons before the cutoff enroll with zero known queries;
    singletons after it are novel queries -- both must survive, Melops is
    ~2.5 images/individual.

    ``same_date_policy`` governs images of an enrolled unit taken on the SAME
    date as its gallery image. On real Melops these are same-handling-session
    near-duplicates that inflate known-query metrics, so the default
    ``"exclude"`` drops them from the query set (the count is recorded in
    ``query_df.attrs["n_same_date_excluded"]`` and must be reported).
    ``"include"`` keeps them as known queries, for measuring the inflation.

    Dates must parse and be non-null: a NaT date would be silently mis-binned,
    so it raises ``ProtocolViolation`` instead.

    Returns ``(gallery_df, query_df)``; ``query_df`` carries ``is_known``.
    """
    _check_input_frame(df)
    if same_date_policy not in ("exclude", "include"):
        raise ProtocolViolation("same_date_policy must be 'exclude' or 'include', got %r"
                                % (same_date_policy,))
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    if df["_date"].isna().any():
        bad = df.loc[df["_date"].isna(), "image_id"].tolist()[:5]
        raise ProtocolViolation("unparseable/missing dates (first ids: %r); "
                                "fix the metadata rather than letting the split guess" % (bad,))
    cutoff = _resolve_cutoff(df["_date"], cutoff_fraction, cutoff_date)
    rng = np.random.default_rng(int(seed))

    gallery_rows = []
    query_rows = []
    n_same_date_excluded = 0
    for (_identity, _side), group in df.groupby(["identity", "side"], sort=True):
        group = group.sort_values(["_date", "image_id"])
        first_date = group["_date"].iloc[0]
        if first_date < cutoff:
            earliest = group[group["_date"] == first_date]
            pick = int(rng.integers(0, len(earliest)))
            gallery_row = earliest.iloc[pick]
            gallery_rows.append(gallery_row)
            rest = group[group["image_id"] != gallery_row["image_id"]]
            if same_date_policy == "exclude":
                same_date = rest["_date"] == gallery_row["_date"]
                n_same_date_excluded += int(same_date.sum())
                rest = rest[~same_date]
            for _, row in rest.iterrows():
                row = row.copy()
                row["is_known"] = True
                query_rows.append(row)
        else:
            for _, row in group.iterrows():
                row = row.copy()
                row["is_known"] = False
                query_rows.append(row)

    gallery_df = pd.DataFrame(gallery_rows).reset_index(drop=True)
    if query_rows:
        query_df = pd.DataFrame(query_rows).reset_index(drop=True)
    else:
        query_df = pd.DataFrame(columns=list(df.columns) + ["is_known"])
    for frame in (gallery_df, query_df):
        if "_date" in frame.columns:
            frame.drop(columns=["_date"], inplace=True)
    if len(gallery_df) == 0:
        raise ProtocolViolation("empty gallery: cutoff excludes every identity")
    _check_split(gallery_df, query_df)
    query_df.attrs["n_same_date_excluded"] = n_same_date_excluded
    return gallery_df, query_df


def _unit_key(frame):
    return list(zip(frame["identity"].tolist(), frame["side"].tolist()))


def _check_split(gallery_df, query_df):
    gallery_ids = set(gallery_df["image_id"])
    if len(gallery_ids) != len(gallery_df):
        raise ProtocolViolation("duplicate image in gallery")
    overlap = gallery_ids.intersection(set(query_df["image_id"]))
    if overlap:
        raise ProtocolViolation("image(s) in both gallery and query: %r" % sorted(overlap)[:3])
    units = _unit_key(gallery_df)
    if len(set(units)) != len(units):
        raise ProtocolViolation("gallery is not one-shot: repeated (identity, side)")
    unit_set = set(units)
    if len(query_df) > 0:
        for _, row in query_df.iterrows():
            key = (row["identity"], row["side"])
            if row["is_known"] and key not in unit_set:
                raise ProtocolViolation("known query %r has no gallery entry" % (key,))
            if (not row["is_known"]) and key in unit_set:
                raise ProtocolViolation("novel query %r leaks into gallery" % (key,))


def cross_orientation_split(df, enroll_side="L", query_side="R", cutoff_fraction=None,
                            cutoff_date=None, seed=0, same_date_policy="exclude"):
    """Enroll one side, query the other. THE ONLY ARM THAT CROSSES SIDES.

    This split ignores the same-side matching rule BY DESIGN -- it measures
    whether identity survives a flank flip. Gallery: earliest ``enroll_side``
    image of each identity whose first ``enroll_side`` sighting precedes the
    cutoff. Queries: all ``query_side`` images; known iff the identity is
    enrolled. Evaluate the result with ``evaluate(..., cross_side=True)``.

    ``same_date_policy`` mirrors ``one_shot_open_set_split``, applied across
    sides: a known query taken on the SAME date as its identity's gallery
    image is the opposite flank of the same handling session (the fish is
    photographed on both sides in one session), so the default ``"exclude"``
    drops it from the query set (count recorded in
    ``query_df.attrs["n_same_date_excluded"]`` and must be reported).
    ``"include"`` keeps them as known queries, for measuring the inflation.
    """
    if enroll_side == query_side:
        raise ValueError("enroll_side and query_side must differ")
    if same_date_policy not in ("exclude", "include"):
        raise ProtocolViolation("same_date_policy must be 'exclude' or 'include', got %r"
                                % (same_date_policy,))
    _check_input_frame(df)
    enroll_df = df[df["side"] == enroll_side]
    if len(enroll_df) == 0:
        raise ProtocolViolation("no images on enroll side %r" % enroll_side)
    gallery_df, _ = one_shot_open_set_split(
        enroll_df, cutoff_fraction=cutoff_fraction, cutoff_date=cutoff_date, seed=seed
    )
    enrolled = set(gallery_df["identity"])
    query_df = df[df["side"] == query_side].copy().reset_index(drop=True)
    if len(query_df) == 0:
        raise ProtocolViolation("no images on query side %r" % query_side)
    query_df["is_known"] = query_df["identity"].isin(enrolled)
    query_dates = pd.to_datetime(query_df["date"])
    if query_dates.isna().any():
        bad = query_df.loc[query_dates.isna(), "image_id"].tolist()[:5]
        raise ProtocolViolation("unparseable/missing dates (first ids: %r); "
                                "fix the metadata rather than letting the split guess" % (bad,))
    n_same_date_excluded = 0
    if same_date_policy == "exclude":
        gallery_date_by_identity = dict(
            zip(gallery_df["identity"], pd.to_datetime(gallery_df["date"]))
        )
        enrolled_date = pd.to_datetime(query_df["identity"].map(gallery_date_by_identity))
        same_date = query_df["is_known"] & (query_dates == enrolled_date)
        n_same_date_excluded = int(same_date.sum())
        query_df = query_df.loc[~same_date].reset_index(drop=True)
        if len(query_df) == 0:
            raise ProtocolViolation(
                "same-date exclusion emptied the query side %r (all %d queries were "
                "same-session opposite flanks)" % (query_side, n_same_date_excluded)
            )
    overlap = set(gallery_df["image_id"]).intersection(set(query_df["image_id"]))
    if overlap:
        raise ProtocolViolation("image(s) in both gallery and query: %r" % sorted(overlap)[:3])
    query_df.attrs["n_same_date_excluded"] = n_same_date_excluded
    return gallery_df, query_df


def _rankdata_average(values):
    """Average ranks (1-based) with tie handling; numpy-only."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _auroc(scores_pos, scores_neg):
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return None
    combined = np.concatenate([scores_pos, scores_neg])
    ranks = _rankdata_average(combined)
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


DEFAULT_REJECTION_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def evaluate(
    gallery_emb,
    gallery_df,
    query_emb,
    query_df,
    cross_side=False,
    rejection_quantiles=DEFAULT_REJECTION_QUANTILES,
):
    """Cosine-similarity retrieval metrics under the open-set protocol.

    Embeddings must be row-aligned with their frames and L2-normalized (cosine
    similarity is then the dot product; this is asserted). Matching is
    restricted to the query's own side unless ``cross_side=True``, in which
    case the gallery must hold exactly one side and every query the other
    (the cross-orientation arm), and match identity ignores side.

    Returns a dict with: n_gallery, n_known, n_novel, rank1, rank5, mAP over
    known queries; open-set novelty AUROC on max-similarity; and a rejection
    curve -- for each quantile of the pooled query max-similarities, Rank-1
    over ALL known queries with rejected-known counted wrong, plus known
    acceptance and novel rejection rates.
    """
    gallery_emb = np.asarray(gallery_emb, dtype=np.float64)
    query_emb = np.asarray(query_emb, dtype=np.float64)
    if gallery_emb.shape[0] != len(gallery_df):
        raise ProtocolViolation("gallery embeddings misaligned with gallery frame")
    if query_emb.shape[0] != len(query_df):
        raise ProtocolViolation("query embeddings misaligned with query frame")
    for emb, label in ((gallery_emb, "gallery"), (query_emb, "query")):
        norms = np.linalg.norm(emb, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ProtocolViolation("%s embeddings are not L2-normalized" % label)
    if "is_known" not in query_df.columns:
        raise ProtocolViolation("query frame missing is_known")
    _check_split_for_eval(gallery_df, query_df, cross_side)

    gallery_sides = gallery_df["side"].to_numpy()
    gallery_identities = gallery_df["identity"].to_numpy()

    n_query = len(query_df)
    max_sims = np.full(n_query, -np.inf)
    ranks = np.full(n_query, np.iinfo(np.int64).max, dtype=np.int64)
    is_known = query_df["is_known"].to_numpy().astype(bool)

    for qi in range(n_query):
        q_side = query_df["side"].iloc[qi]
        if cross_side:
            cols = np.arange(len(gallery_df))
        else:
            cols = np.flatnonzero(gallery_sides == q_side)
        if len(cols) == 0:
            if is_known[qi]:
                raise ProtocolViolation(
                    "known query on side %r has no same-side gallery" % q_side
                )
            continue  # novel query on an un-enrolled side: max_sim stays -inf
        sims = query_emb[qi] @ gallery_emb[cols].T
        max_sims[qi] = float(sims.max())
        if is_known[qi]:
            q_identity = query_df["identity"].iloc[qi]
            match = gallery_identities[cols] == q_identity
            if not cross_side:
                match &= gallery_sides[cols] == q_side
            true_pos = np.flatnonzero(match)
            if len(true_pos) != 1:
                raise ProtocolViolation(
                    "known query %r must have exactly one gallery match, got %d"
                    % (q_identity, len(true_pos))
                )
            true_sim = sims[true_pos[0]]
            ranks[qi] = int((sims > true_sim).sum()) + 1

    known_ranks = ranks[is_known]
    n_known = int(is_known.sum())
    n_novel = int((~is_known).sum())
    if n_known > 0:
        rank1 = float((known_ranks == 1).mean())
        rank5 = float((known_ranks <= 5).mean())
        mean_ap = float((1.0 / known_ranks).mean())  # one relevant item: AP = 1/rank
    else:
        rank1 = rank5 = mean_ap = None

    auroc = _auroc(max_sims[is_known], max_sims[~is_known])

    curve = []
    finite = max_sims[np.isfinite(max_sims)]
    for q in rejection_quantiles:
        if len(finite) == 0:
            break
        threshold = float(np.quantile(finite, q))
        accepted = max_sims >= threshold
        known_accepted = accepted & is_known
        entry = {
            "quantile": float(q),
            "threshold": threshold,
            "known_acceptance_rate": float(known_accepted.sum() / n_known) if n_known else None,
            "novel_rejection_rate": float((~accepted & ~is_known).sum() / n_novel) if n_novel else None,
            "rank1_at_threshold": float(((known_ranks == 1) & accepted[is_known]).sum() / n_known)
            if n_known
            else None,
        }
        curve.append(entry)

    return {
        "n_gallery": int(len(gallery_df)),
        "n_known": n_known,
        "n_novel": n_novel,
        "rank1": rank1,
        "rank5": rank5,
        "mAP": mean_ap,
        "open_set_auroc": auroc,
        "rejection_curve": curve,
    }


def _check_split_for_eval(gallery_df, query_df, cross_side):
    overlap = set(gallery_df["image_id"]).intersection(set(query_df["image_id"]))
    if overlap:
        raise ProtocolViolation("image(s) in both gallery and query: %r" % sorted(overlap)[:3])
    if cross_side:
        g_sides = set(gallery_df["side"])
        q_sides = set(query_df["side"])
        if len(g_sides) != 1 or len(q_sides) != 1 or g_sides == q_sides:
            raise ProtocolViolation(
                "cross_side evaluation requires one gallery side and the other "
                "query side, got gallery=%r query=%r" % (sorted(g_sides), sorted(q_sides))
            )
        ids = gallery_df["identity"]
        if ids.duplicated().any():
            raise ProtocolViolation("cross-side gallery is not one-shot per identity")
    else:
        units = _unit_key(gallery_df)
        if len(set(units)) != len(units):
            raise ProtocolViolation("gallery is not one-shot: repeated (identity, side)")
        unit_set = set(units)
        for _, row in query_df.iterrows():
            key = (row["identity"], row["side"])
            if row["is_known"] and key not in unit_set:
                raise ProtocolViolation("known query %r has no gallery entry" % (key,))
            if (not row["is_known"]) and key in unit_set:
                raise ProtocolViolation("novel query %r leaks into gallery" % (key,))
