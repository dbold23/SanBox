"""Behavioral tests for the run-2 fine-tune arm (CPU, tiny backbone, fast).

Everything runs on the synthetic corpus with a random-init tiny timm model
(``resnet10t`` at 64 px): huggingface.co is egress-blocked here, so no
pretrained weights are ever requested.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

import embedders  # noqa: E402
import finetune  # noqa: E402
import melops_data  # noqa: E402
import protocol  # noqa: E402

TINY = dict(
    backbone="resnet10t",
    bbox="body",
    train_frac=0.6,
    batch_size=8,
    lr=1e-2,
    backbone_lr=1e-2,  # random init: the backbone must learn as fast as the head
    img_size=64,
    pretrained=False,
    device="cpu",
    seed=0,
)

# Micro-run epochs: 2 epochs of FROM-SCRATCH ArcFace (no pretrained weights
# reach this sandbox) is mid-collapse and can land below the random-init
# baseline depending on seed; 8 epochs converges (final loss ~8 vs ~21 at
# init) and still trains in ~10 s on CPU, well inside the suite budget.
MICRO_EPOCHS = 8


@pytest.fixture(scope="session")
def ft_corpus(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("ft_corpus"))
    melops_data.make_synthetic(
        root, n_individuals=36, images_per_individual_dist=(4, 5), seed=23
    )
    return root


@pytest.fixture(scope="session")
def trained(ft_corpus, tmp_path_factory):
    """Micro-run on the synthetic train split (tiny backbone, 64 px)."""
    out = str(tmp_path_factory.mktemp("ft_trained"))
    return finetune.train_finetune(ft_corpus, out, epochs=MICRO_EPOCHS, **TINY)


@pytest.fixture(scope="session")
def untrained(ft_corpus, tmp_path_factory):
    """Same seed, same architecture, ZERO epochs: the random-init baseline."""
    out = str(tmp_path_factory.mktemp("ft_untrained"))
    return finetune.train_finetune(ft_corpus, out, epochs=0, **TINY)


# ---------------------------------------------------------------------------
# 1. Identity-disjoint split
# ---------------------------------------------------------------------------


def test_split_identities_disjoint_and_complete(ft_corpus):
    df = melops_data.load_melops(ft_corpus)
    train_df, eval_df = finetune.split_identities(df, train_frac=0.6, seed=0)
    train_ids = set(train_df["identity"])
    eval_ids = set(eval_df["identity"])
    assert len(train_ids & eval_ids) == 0  # ZERO identity overlap
    assert train_ids | eval_ids == set(df["identity"])  # nothing dropped
    assert len(train_df) + len(eval_df) == len(df)  # every image kept
    n_ids = df["identity"].nunique()
    assert len(train_ids) == int(round(0.6 * n_ids))
    # both flanks of one fish travel together: sides never split an identity
    for ident in train_ids:
        assert not (eval_df["identity"] == ident).any()


def test_split_identities_deterministic_and_seed_sensitive(ft_corpus):
    df = melops_data.load_melops(ft_corpus)
    a1, _ = finetune.split_identities(df, seed=7)
    a2, _ = finetune.split_identities(df, seed=7)
    b1, _ = finetune.split_identities(df, seed=8)
    assert set(a1["identity"]) == set(a2["identity"])
    assert set(a1["identity"]) != set(b1["identity"])


def test_split_identities_rejects_bad_frac(ft_corpus):
    df = melops_data.load_melops(ft_corpus)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            finetune.split_identities(df, train_frac=bad)


def test_eval_split_feeds_protocol_unchanged(ft_corpus):
    """The eval side goes through the EXISTING protocol split, untouched."""
    df = melops_data.load_melops(ft_corpus)
    train_df, eval_df = finetune.split_identities(df, train_frac=0.6, seed=0)
    gallery_df, query_df = protocol.one_shot_open_set_split(
        eval_df, cutoff_fraction=0.5, seed=0
    )
    seen = set(gallery_df["identity"]) | set(query_df["identity"])
    assert len(seen & set(train_df["identity"])) == 0


# ---------------------------------------------------------------------------
# 2. ArcFace training
# ---------------------------------------------------------------------------


def test_one_training_step_decreases_arcface_loss(tmp_path):
    """One optimizer step on an 8-identity corpus lowers the ArcFace loss."""
    root = str(tmp_path / "corpus8")
    df = melops_data.make_synthetic(
        root, n_individuals=8, images_per_individual_dist=(2, 3), seed=5
    )
    finetune._seed_everything(0)
    units, labels = finetune._class_units(df)
    model = timm.create_model("resnet10t", pretrained=False, num_classes=0)
    head = finetune.ArcMarginProduct(model.num_features, len(units), s=30.0, m=0.5)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=1e-3
    )
    crops = [
        finetune._preprocess(melops_data.load_crop(root, row), 64)
        for _, row in df.iterrows()
    ]
    x = torch.from_numpy(np.stack(crops)).float()
    y = torch.from_numpy(labels)

    def loss_now():
        return torch.nn.functional.cross_entropy(head(model(x), y), y)

    model.train()
    loss0 = loss_now()
    optimizer.zero_grad()
    loss0.backward()
    optimizer.step()
    with torch.no_grad():
        loss1 = loss_now()
    assert float(loss1.detach()) < float(loss0.detach())


def test_train_log_records_per_epoch_loss(trained):
    import json

    with open(trained["train_log"]) as f:
        log = json.load(f)
    assert [e["epoch"] for e in log["epochs"]] == list(range(MICRO_EPOCHS))
    assert all(np.isfinite(e["mean_loss"]) for e in log["epochs"])
    # training must actually descend, not just log numbers
    assert log["epochs"][-1]["mean_loss"] < log["epochs"][0]["mean_loss"]
    assert log["config"]["hflip"] is False  # mirroring would alias flanks
    assert log["config"]["n_train_classes"] >= 2


# ---------------------------------------------------------------------------
# 3. Checkpoint round-trip / embedder bridge
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_l2_normed_deterministic(ft_corpus, trained):
    df = melops_data.load_melops(ft_corpus)
    crops = [melops_data.load_crop(ft_corpus, r) for _, r in df.head(5).iterrows()]
    emb_a = finetune.FinetunedEmbedder(trained["checkpoint"], device="cpu")
    emb_b = finetune.FinetunedEmbedder(trained["checkpoint"], device="cpu")
    va = emb_a.embed(crops)
    vb = emb_b.embed(crops)
    assert va.dtype == np.float64 and va.shape[0] == 5
    assert np.allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-9)
    assert np.array_equal(va, vb)  # save -> load -> embed is deterministic
    assert np.array_equal(va, emb_a.embed(crops))  # and idempotent


def test_spec_dispatch_and_factory_hook(trained):
    spec = "finetuned:" + trained["checkpoint"]
    for maker in (finetune.get_embedder_from_spec, embedders.get_embedder):
        emb = maker(spec)
        assert isinstance(emb, finetune.FinetunedEmbedder)
    # non-prefixed specs fall through to the ordinary factory
    assert isinstance(finetune.get_embedder_from_spec("hist"), embedders.HistEmbedder)
    with pytest.raises(ValueError):
        finetune.get_embedder_from_spec("finetuned:/no/such/checkpoint.pt")


# ---------------------------------------------------------------------------
# 4. End-to-end micro-run: fine-tuning must demonstrably help
# ---------------------------------------------------------------------------


def _eval_rank1(root, ckpt_path, eval_df):
    gallery_df, query_df = protocol.one_shot_open_set_split(
        eval_df, cutoff_fraction=0.5, seed=0
    )
    emb = finetune.FinetunedEmbedder(ckpt_path, device="cpu", batch_size=16)
    g = emb.embed([melops_data.load_crop(root, r) for _, r in gallery_df.iterrows()])
    q = emb.embed([melops_data.load_crop(root, r) for _, r in query_df.iterrows()])
    return protocol.evaluate(g, gallery_df, q, query_df)


def test_finetuned_beats_random_init_on_heldout_identities(
    ft_corpus, trained, untrained
):
    # identical split by construction (same seed / frac)
    assert set(trained["eval_df"]["identity"]) == set(untrained["eval_df"]["identity"])
    res_ft = _eval_rank1(ft_corpus, trained["checkpoint"], trained["eval_df"])
    res_rand = _eval_rank1(ft_corpus, untrained["checkpoint"], untrained["eval_df"])
    assert res_ft["n_known"] >= 10  # enough known queries to mean anything
    print(
        "\nmicro-run (held-out identities): fine-tuned Rank-1=%.3f mAP=%.3f | "
        "random-init Rank-1=%.3f mAP=%.3f | n_known=%d n_novel=%d gallery=%d"
        % (
            res_ft["rank1"],
            res_ft["mAP"],
            res_rand["rank1"],
            res_rand["mAP"],
            res_ft["n_known"],
            res_ft["n_novel"],
            res_ft["n_gallery"],
        )
    )
    assert res_ft["rank1"] > res_rand["rank1"]  # fine-tuning must help


def test_split_rejects_duplicate_path_and_null_identity():
    import pandas as pd
    import pytest
    from protocol import ProtocolViolation
    import finetune

    base = pd.DataFrame({
        "image_id": ["i1", "i2", "i3"],
        "identity": ["A", "B", "C"],
        "date": ["2018-01-01"] * 3,
        "side": ["L"] * 3,
        "path": ["p1.png", "p2.png", "p3.png"],
    })
    dup = base.copy()
    dup.loc[2, "path"] = "p1.png"  # same pixels under two identities
    with pytest.raises(ProtocolViolation, match="duplicate path"):
        finetune.split_identities(dup, seed=9)

    nan = base.copy()
    nan.loc[1, "identity"] = None
    with pytest.raises(ProtocolViolation, match="null identity"):
        finetune.split_identities(nan, seed=0)


def test_run_ablation_excludes_checkpoint_train_identities(tmp_path):
    import json
    import pandas as pd
    import pytest
    import torch
    import run_ablation
    from protocol import ProtocolViolation

    ckpt = tmp_path / "ckpt.pt"
    torch.save({"classes": [["A", "L"], ["A", "R"], ["B", "L"]]}, str(ckpt))
    df = pd.DataFrame({
        "image_id": ["i1", "i2", "i3", "i4"],
        "identity": ["A", "B", "C", "D"],
        "date": ["2018-01-01"] * 4,
        "side": ["L"] * 4,
        "path": ["p1.png", "p2.png", "p3.png", "p4.png"],
    })
    out, n_ids, n_rows = run_ablation._exclude_train_identities(df, "finetuned:%s" % ckpt)
    assert set(out["identity"]) == {"C", "D"}
    assert (n_ids, n_rows) == (2, 2)

    # a non-finetuned backbone is a no-op
    same, z1, z2 = run_ablation._exclude_train_identities(df, "hist")
    assert len(same) == 4 and (z1, z2) == (0, 0)

    # a checkpoint with no identity record is refused, not trusted
    empty = tmp_path / "empty.pt"
    torch.save({"classes": []}, str(empty))
    with pytest.raises(ProtocolViolation, match="no training-identity record"):
        run_ablation._exclude_train_identities(df, "finetuned:%s" % empty)

    # excluding everything is refused
    allids = tmp_path / "all.pt"
    torch.save({"classes": [[i, "L"] for i in "ABCD"]}, str(allids))
    with pytest.raises(ProtocolViolation, match="removed every row"):
        run_ablation._exclude_train_identities(df, "finetuned:%s" % allids)


def test_backbone_arg_accepts_finetuned_prefix():
    import argparse
    import pytest
    import run_ablation

    assert run_ablation._backbone_arg("hist") == "hist"
    assert run_ablation._backbone_arg("finetuned:/x/ckpt.pt") == "finetuned:/x/ckpt.pt"
    with pytest.raises(argparse.ArgumentTypeError):
        run_ablation._backbone_arg("resnet50")
