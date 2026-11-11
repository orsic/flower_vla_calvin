"""scripts/simulate_modality_sampling.py must reproduce the training code's sampling
(flower/models/networks/modality_dropout.py) exactly, or in distribution where it draws
its own randomness."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import simulate_modality_sampling as sim  # noqa: E402
from flower.models.networks import modality_dropout as md  # noqa: E402


class _FixedDirichlet:
    props: torch.Tensor

    def __init__(self, alphas):
        pass

    def sample(self, shape):
        return self.props


@pytest.mark.parametrize(
    "avail, keep_fraction",
    [
        ([[17, 17, 37], [17, 17, 33], [17, 17, 44]], 0.5),  # LIBERO: images cap the budget
        ([[17, 17, 40]] * 3, 0.75),
        ([[50, 50, 30], [50, 50, 37], [50, 50, 28]], 0.25),
        ([[50, 50, 34]] * 3, 0.5),
    ],
)
def test_dirichlet_counts_match_training_code_for_the_same_proportions(monkeypatch, avail, keep_fraction):
    props = np.array([[0.9, 0.06, 0.04], [0.02, 0.48, 0.5], [0.3, 0.33, 0.37]])
    _FixedDirichlet.props = torch.tensor(props, dtype=torch.float32)
    monkeypatch.setattr(md, "Dirichlet", _FixedDirichlet)
    expected = md.sample_token_budget(torch.tensor(avail), keep_fraction, torch.ones(3)).numpy()
    got = sim.dirichlet_counts(np.array(avail), keep_fraction, np.random.default_rng(0), props=props.astype(np.float32))
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("strategy", ["dirichlet", "uniform"])
def test_budget_is_exact_and_capped(strategy):
    rng = np.random.default_rng(1)
    avail = np.array([[17, 17, 33], [17, 17, 44]] * 500)
    fn = sim.dirichlet_counts if strategy == "dirichlet" else lambda a, f, r: sim.uniform_counts(a, f, r)
    counts = fn(avail, 0.5, rng)
    assert (counts.sum(axis=1) == round(0.5 * 67)).all()
    assert (counts <= avail).all() and (counts >= 0).all()


def test_uniform_counts_match_training_code_in_distribution():
    avail = np.array([[50, 50, 32]] * 20000)
    got = sim.uniform_counts(avail, 0.5, np.random.default_rng(2))
    torch.manual_seed(2)
    ref = md.sample_uniform_counts(torch.tensor(avail), 0.5).numpy()
    np.testing.assert_allclose(got.mean(axis=0), ref.mean(axis=0), atol=0.1)
    np.testing.assert_allclose(got.std(axis=0), ref.std(axis=0), atol=0.1)


def test_dirichlet_counts_match_training_code_in_distribution():
    avail = np.array([[17, 17, 37]] * 20000)
    got = sim.dirichlet_counts(avail, 0.5, np.random.default_rng(3))
    torch.manual_seed(3)
    ref = md.sample_token_budget(torch.tensor(avail), 0.5, torch.ones(3)).numpy()
    np.testing.assert_allclose(got.mean(axis=0), ref.mean(axis=0), atol=0.15)
    np.testing.assert_allclose((got == 0).mean(axis=0), (ref == 0).mean(axis=0), atol=0.01)


@pytest.mark.parametrize("n", [3, 4])
def test_subset_probabilities_match_training_code_and_closed_form(n):
    got = sim.subset_masks(40000, n, np.random.default_rng(4))
    torch.manual_seed(4)
    ref = md.sample_modality_subsets(40000, n, torch.device("cpu")).numpy()
    assert got.any(axis=1).all()
    size = got.sum(axis=1)
    for k in range(1, n + 1):
        n_subsets = {1: n, 2: n * (n - 1) // 2, 3: n * (n - 1) * (n - 2) // 6, 4: 1}[k]
        closed = n_subsets * 2**k / (3**n - 1)
        assert (size == k).mean() == pytest.approx(closed, abs=0.01)
        assert (size == k).mean() == pytest.approx((ref.sum(axis=1) == k).mean(), abs=0.01)
    np.testing.assert_allclose(got.mean(axis=0), ref.mean(axis=0), atol=0.01)


def test_subset_with_proprio_keeps_each_input_whole_or_not_at_all():
    ds = sim.DATASETS["calvin"]
    exp = sim.Experiment("subset", "subset", proprio=True)
    avail, kept = sim.simulate(ds, exp, 40000, np.random.default_rng(5))
    for s in sim.summarize(avail, kept):
        assert s["p_partial"] == 0
        assert s["p_full"] == pytest.approx(54 / 80, abs=0.01)  # 16 + 3*8 + 3*4 + 2 of 80


def test_proprio_gate_and_no_dropout():
    ds = sim.DATASETS["mimicgen"]
    rng = np.random.default_rng(6)
    avail, kept = sim.simulate(ds, sim.Experiment("d", "dirichlet", 0.5, True, 0.5), 40000, rng)
    assert kept[:, 3].mean() == pytest.approx(0.5, abs=0.01)
    avail, kept = sim.simulate(ds, sim.Experiment("n", "none"), 1000, rng)
    np.testing.assert_array_equal(avail, kept)
    assert [s["modality"] for s in sim.summarize(avail, kept)] == ["static", "wrist", "language"]
