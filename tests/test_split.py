import numpy as np
import pytest

from src.split import assert_no_speaker_leakage, make_folds


def _toy(n_spk=12, per=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"s{i}" for i in range(n_spk)], per)
    y = np.repeat(np.arange(n_spk) % 2, per)
    perm = rng.permutation(len(y))
    return y[perm], groups[perm]


def test_no_speaker_in_train_and_test():
    y, groups = _toy()
    folds = make_folds(y, groups, n_splits=5, seed=42)
    assert len(folds) == 5
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
        assert set(y[te]) == {0, 1}  # stratified: both classes in every test fold
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered) == list(range(len(y)))


def test_folds_reduced_when_few_speakers():
    y, groups = _toy(n_spk=6)
    assert len(make_folds(y, groups, n_splits=5)) == 3


def test_leakage_detected():
    groups = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError):
        assert_no_speaker_leakage([(np.array([0, 2]), np.array([1, 3]))], groups)
