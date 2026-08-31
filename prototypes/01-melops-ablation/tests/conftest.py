from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import melops_data  # noqa: E402


@pytest.fixture(scope="session")
def distributed_corpus(tmp_path_factory):
    """Identity signal on both head and body regions."""
    root = str(tmp_path_factory.mktemp("distributed"))
    melops_data.make_synthetic(root, n_individuals=40, seed=11, head_signal=1.0, body_signal=1.0)
    return root


@pytest.fixture(scope="session")
def head_corpus(tmp_path_factory):
    """Identity signal concentrated in the head third; body uninformative."""
    root = str(tmp_path_factory.mktemp("head_only"))
    melops_data.make_synthetic(root, n_individuals=40, seed=11, head_signal=1.0, body_signal=0.0)
    return root
