"""Unit tests for datasets/planning.py"""

import random
import pickle
import sys
sys.path.insert(0, ".")

from datasets.planning import SampleSpec, SamplePlanner, serialize_sample_spec, deserialize_sample_spec


class MockDatasetSampler:
    """Minimal mock for DatasetSampler."""
    def __init__(self, sequences, clip_len=8):
        self.valid_sequences = sequences
        self.clip_len = clip_len
        self._active_sequence = None

    def sample(self, rng):
        seq = rng.choice(self.valid_sequences)
        num_frames = 100
        start = rng.randint(0, num_frames - self.clip_len)
        return seq, list(range(start, start + self.clip_len))

    def reset_locality_state(self):
        self._active_sequence = None


class MockMixtureSampler:
    """Minimal mock for MixtureSampler."""
    def __init__(self, dataset_samplers, dataset_weights=None):
        self.samplers = dataset_samplers
        n = len(dataset_samplers)
        self.dataset_probs = [1.0/n] * n if dataset_weights is None else [w/sum(dataset_weights) for w in dataset_weights]
        self._active_dataset_idx = None
        self._remaining_dataset_uses = 0
        self.dataset_locality_size = 2

    def sample(self, rng):
        # Simple: pick dataset weighted, then sample from it
        dataset_idx = rng.choices(range(len(self.samplers)), weights=self.dataset_probs, k=1)[0]
        seq_name, frame_indices = self.samplers[dataset_idx].sample(rng)
        return dataset_idx, seq_name, frame_indices

    def reset_locality_state(self, dataset_idx=None):
        self._active_dataset_idx = None
        self._remaining_dataset_uses = 0
        if dataset_idx is None:
            for s in self.samplers:
                s.reset_locality_state()


def _make_planner():
    ds0 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    ds1 = MockDatasetSampler(["seq_x", "seq_y"], clip_len=8)
    mixture = MockMixtureSampler([ds0, ds1], dataset_weights=[0.7, 0.3])
    return SamplePlanner(mixture_sampler=mixture, seed=42, epoch=0)


def test_plan_determinism():
    """Same planner, same seed → same plan."""
    planner1 = _make_planner()
    plan1 = planner1.generate_plan(start_index=0, count=20)

    planner2 = _make_planner()
    plan2 = planner2.generate_plan(start_index=0, count=20)

    for s1, s2 in zip(plan1, plan2):
        assert s1.dataset_idx == s2.dataset_idx, f"dataset_idx mismatch at {s1.sample_index}"
        assert s1.sequence_name == s2.sequence_name, f"sequence_name mismatch at {s1.sample_index}"
        assert s1.frame_indices == s2.frame_indices, f"frame_indices mismatch at {s1.sample_index}"
        assert s1.rng_state == s2.rng_state, f"rng_state mismatch at {s1.sample_index}"
    print("✓ test_plan_determinism passed")


def test_plan_different_seeds():
    """Different seeds → different plans."""
    ds0 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    mixture = MockMixtureSampler([ds0])

    planner1 = SamplePlanner(mixture_sampler=mixture, seed=42, epoch=0)
    plan1 = planner1.generate_plan(start_index=0, count=10)

    ds0_2 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    mixture2 = MockMixtureSampler([ds0_2])
    planner2 = SamplePlanner(mixture_sampler=mixture2, seed=123, epoch=0)
    plan2 = planner2.generate_plan(start_index=0, count=10)

    any_different = any(
        s1.frame_indices != s2.frame_indices for s1, s2 in zip(plan1, plan2)
    )
    assert any_different, "Different seeds should produce different plans"
    print("✓ test_plan_different_seeds passed")


def test_sample_spec_serialization():
    """SampleSpec can be serialized and deserialized."""
    rng = random.Random(42)
    spec = SampleSpec(
        dataset_idx=1,
        sequence_name="test_seq",
        frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        rng_state=rng.getstate(),
        sample_index=99,
    )

    data = serialize_sample_spec(spec)
    assert isinstance(data, bytes)

    spec2 = deserialize_sample_spec(data)
    assert spec2.dataset_idx == spec.dataset_idx
    assert spec2.sequence_name == spec.sequence_name
    assert spec2.frame_indices == spec.frame_indices
    assert spec2.rng_state == spec.rng_state
    assert spec2.sample_index == spec.sample_index
    print("✓ test_sample_spec_serialization passed")


def test_rng_state_reproducibility():
    """Captured RNG state allows reproducing subsequent random calls."""
    planner = _make_planner()
    plan = planner.generate_plan(start_index=0, count=5)

    for spec in plan:
        # Restore RNG state and generate some values
        rng1 = random.Random()
        rng1.setstate(spec.rng_state)
        vals1 = [rng1.random() for _ in range(10)]

        rng2 = random.Random()
        rng2.setstate(spec.rng_state)
        vals2 = [rng2.random() for _ in range(10)]

        assert vals1 == vals2, "RNG state restore should produce identical values"
    print("✓ test_rng_state_reproducibility passed")


def test_plan_count_and_indices():
    """Plan contains correct number of specs with correct indices."""
    planner = _make_planner()
    plan = planner.generate_plan(start_index=100, count=15)

    assert len(plan) == 15
    for i, spec in enumerate(plan):
        assert spec.sample_index == 100 + i
    print("✓ test_plan_count_and_indices passed")


if __name__ == "__main__":
    test_plan_determinism()
    test_plan_different_seeds()
    test_sample_spec_serialization()
    test_rng_state_reproducibility()
    test_plan_count_and_indices()
    print("\nAll planning tests passed!")
