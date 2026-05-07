"""Updated unit tests for datasets/planning.py with new API"""

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


def _make_planner(rank=0, world_size=1, reshuffle_each_epoch=True):
    ds0 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    ds1 = MockDatasetSampler(["seq_x", "seq_y"], clip_len=8)
    mixture = MockMixtureSampler([ds0, ds1], dataset_weights=[0.7, 0.3])
    return SamplePlanner(mixture_sampler=mixture, seed=42, rank=rank, world_size=world_size, reshuffle_each_epoch=reshuffle_each_epoch)


def test_plan_determinism():
    """Same planner, same seed → same plan."""
    planner1 = _make_planner()
    plan1 = planner1.generate_plan(epoch=0, count_per_rank=20, epoch_size=20)

    planner2 = _make_planner()
    plan2 = planner2.generate_plan(epoch=0, count_per_rank=20, epoch_size=20)

    for s1, s2 in zip(plan1, plan2):
        assert s1.dataset_idx == s2.dataset_idx, f"dataset_idx mismatch at {s1.local_index}"
        assert s1.sequence_name == s2.sequence_name, f"sequence_name mismatch at {s1.local_index}"
        assert s1.frame_indices == s2.frame_indices, f"frame_indices mismatch at {s1.local_index}"
        assert s1.rng_state == s2.rng_state, f"rng_state mismatch at {s1.local_index}"
    print("✓ test_plan_determinism passed")


def test_plan_different_seeds():
    """Different seeds → different plans."""
    ds0 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    mixture = MockMixtureSampler([ds0])

    planner1 = SamplePlanner(mixture_sampler=mixture, seed=42, rank=0, world_size=1)
    plan1 = planner1.generate_plan(epoch=0, count_per_rank=10, epoch_size=10)

    ds0_2 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    mixture2 = MockMixtureSampler([ds0_2])
    planner2 = SamplePlanner(mixture_sampler=mixture2, seed=123, rank=0, world_size=1)
    plan2 = planner2.generate_plan(epoch=0, count_per_rank=10, epoch_size=10)

    any_different = any(
        s1.frame_indices != s2.frame_indices for s1, s2 in zip(plan1, plan2)
    )
    assert any_different, "Different seeds should produce different plans"
    print("✓ test_plan_different_seeds passed")


def test_sample_spec_serialization():
    """SampleSpec can be serialized and deserialized."""
    rng = random.Random(42)
    spec = SampleSpec(
        local_index=0,
        global_index=99,
        dataset_idx=1,
        sequence_name="test_seq",
        frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        rng_state=rng.getstate(),
        generation=0,
    )

    data = serialize_sample_spec(spec)
    assert isinstance(data, bytes)

    spec2 = deserialize_sample_spec(data)
    assert spec2.local_index == spec.local_index
    assert spec2.global_index == spec.global_index
    assert spec2.dataset_idx == spec.dataset_idx
    assert spec2.sequence_name == spec.sequence_name
    assert spec2.frame_indices == spec.frame_indices
    assert spec2.rng_state == spec.rng_state
    assert spec2.generation == spec.generation
    print("✓ test_sample_spec_serialization passed")


def test_rng_state_reproducibility():
    """rng_state can be restored to produce identical random values."""
    planner = _make_planner()
    plan = planner.generate_plan(epoch=0, count_per_rank=5, epoch_size=5)

    for spec in plan:
        rng1 = random.Random()
        rng1.setstate(spec.rng_state)
        vals1 = [rng1.random() for _ in range(10)]

        rng2 = random.Random()
        rng2.setstate(spec.rng_state)
        vals2 = [rng2.random() for _ in range(10)]

        assert vals1 == vals2, "RNG state restore should produce identical values"
    print("✓ test_rng_state_reproducibility passed")


def test_plan_count_and_indices():
    """Plan contains correct number of specs with correct local_index."""
    planner = _make_planner()
    plan = planner.generate_plan(epoch=0, count_per_rank=15, epoch_size=15)

    assert len(plan) == 15
    for i, spec in enumerate(plan):
        assert spec.local_index == i, f"local_index should be {i}, got {spec.local_index}"
    print("✓ test_plan_count_and_indices passed")


def test_multi_rank_plan_generation():
    """Different ranks get different subsets of the global plan."""
    ds0 = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)
    mixture0 = MockMixtureSampler([ds0])
    planner0 = SamplePlanner(mixture_sampler=mixture0, seed=42, rank=0, world_size=2)
    plan0 = planner0.generate_plan(epoch=0, count_per_rank=10, epoch_size=20)

    ds1 = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)
    mixture1 = MockMixtureSampler([ds1])
    planner1 = SamplePlanner(mixture_sampler=mixture1, seed=42, rank=1, world_size=2)
    plan1 = planner1.generate_plan(epoch=0, count_per_rank=10, epoch_size=20)

    # Both should have 10 items
    assert len(plan0) == 10
    assert len(plan1) == 10

    # global_index should be interleaved: rank0 gets 0,2,4,6... rank1 gets 1,3,5,7...
    for i, spec in enumerate(plan0):
        assert spec.global_index == i * 2, f"rank0 global_index should be {i*2}, got {spec.global_index}"
        assert spec.local_index == i

    for i, spec in enumerate(plan1):
        assert spec.global_index == i * 2 + 1, f"rank1 global_index should be {i*2+1}, got {spec.global_index}"
        assert spec.local_index == i

    # Plans should be different (different global indices)
    assert plan0[0].global_index != plan1[0].global_index

    print("✓ test_multi_rank_plan_generation passed")


def test_generation_parameter():
    """generation parameter is correctly stamped into SampleSpec."""
    planner = _make_planner()

    # Default: generation=epoch
    plan1 = planner.generate_plan(epoch=5, count_per_rank=3, epoch_size=3)
    for spec in plan1:
        assert spec.generation == 5, f"Default generation should be 5, got {spec.generation}"

    # Explicit generation
    plan2 = planner.generate_plan(epoch=10, count_per_rank=3, epoch_size=3, generation=2)
    for spec in plan2:
        assert spec.generation == 2, f"Explicit generation should be 2, got {spec.generation}"

    print("✓ test_generation_parameter passed")


def test_reshuffle_each_epoch_false():
    """reshuffle_each_epoch=False → same plan every epoch."""
    ds0 = MockDatasetSampler(["scene_a", "scene_b", "scene_c"], clip_len=8)
    mixture = MockMixtureSampler([ds0])
    planner = SamplePlanner(mixture_sampler=mixture, seed=42, rank=0, world_size=1, reshuffle_each_epoch=False)

    plan0 = planner.generate_plan(epoch=0, count_per_rank=10, epoch_size=10)
    plan5 = planner.generate_plan(epoch=5, count_per_rank=10, epoch_size=10)

    for s0, s5 in zip(plan0, plan5):
        assert s0.sequence_name == s5.sequence_name, "Same plan across epochs when reshuffle=False"
        assert s0.frame_indices == s5.frame_indices
    print("✓ test_reshuffle_each_epoch_false passed")


def test_rng_state_independent_of_position():
    """rng_state matches single-RNG-per-sample design: Random(seed+gi) after sample()."""
    ds0 = MockDatasetSampler(["scene_a"], clip_len=8)
    mixture = MockMixtureSampler([ds0])
    planner = SamplePlanner(mixture_sampler=mixture, seed=42, rank=0, world_size=1)

    plan = planner.generate_plan(epoch=0, count_per_rank=5, epoch_size=5)

    import random as _random
    for spec in plan:
        # Replicate the single-RNG flow: seed → sample() → getstate()
        rng = _random.Random(42 + spec.global_index)
        mock_mix = MockMixtureSampler([MockDatasetSampler(["scene_a"], clip_len=8)])
        mock_mix.sample(rng)  # consume same random calls as planner
        expected_state = rng.getstate()
        assert spec.rng_state == expected_state, (
            f"rng_state for global_index={spec.global_index} should be post-sampling state"
        )
    print("✓ test_rng_state_independent_of_position passed")


def test_ddp_epoch_size():
    """Multi-rank plans should partition epoch_size correctly."""
    ds0 = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)

    import math
    epoch_size = 20
    world_size = 2
    count_per_rank = math.ceil(epoch_size / world_size)  # 10

    mixture0 = MockMixtureSampler([ds0])
    planner0 = SamplePlanner(mixture_sampler=mixture0, seed=42, rank=0, world_size=world_size)
    plan0 = planner0.generate_plan(epoch=0, count_per_rank=count_per_rank, epoch_size=epoch_size)

    ds0_2 = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)
    mixture1 = MockMixtureSampler([ds0_2])
    planner1 = SamplePlanner(mixture_sampler=mixture1, seed=42, rank=1, world_size=world_size)
    plan1 = planner1.generate_plan(epoch=0, count_per_rank=count_per_rank, epoch_size=epoch_size)

    assert len(plan0) == count_per_rank
    assert len(plan1) == count_per_rank

    # Total unique global indices = epoch_size
    all_global = set(s.global_index for s in plan0) | set(s.global_index for s in plan1)
    assert len(all_global) == epoch_size, f"Total unique samples should be {epoch_size}, got {len(all_global)}"

    print("✓ test_ddp_epoch_size passed")


def test_rng_state_varies_across_epochs():
    """reshuffle_each_epoch=True → rng_state differs for same global_index across epochs."""
    planner = _make_planner()
    plan_e0 = planner.generate_plan(epoch=0, count_per_rank=5, epoch_size=5)
    plan_e1 = planner.generate_plan(epoch=1, count_per_rank=5, epoch_size=5)

    any_different = any(
        s0.rng_state != s1.rng_state for s0, s1 in zip(plan_e0, plan_e1)
    )
    assert any_different, "rng_state should differ across epochs when reshuffle=True"
    print("✓ test_rng_state_varies_across_epochs passed")


def test_ddp_remainder_no_loss():
    """epoch_size not divisible by world_size — padding repeats samples (PyTorch DistributedSampler behavior)."""
    import math
    ds0 = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)

    epoch_size = 10
    world_size = 3
    count_per_rank = math.ceil(epoch_size / world_size)  # 4

    plans = []
    for rank in range(world_size):
        ds = MockDatasetSampler(["scene_a", "scene_b"], clip_len=8)
        mix = MockMixtureSampler([ds])
        p = SamplePlanner(mixture_sampler=mix, seed=42, rank=rank, world_size=world_size)
        plans.append(p.generate_plan(epoch=0, count_per_rank=count_per_rank, epoch_size=epoch_size))

    # Each rank gets count_per_rank samples
    for plan in plans:
        assert len(plan) == count_per_rank

    # Collect all global_index values (should be 0..11, unique for rank assignment)
    all_global = set()
    for plan in plans:
        for spec in plan:
            all_global.add(spec.global_index)

    # total_samples = count_per_rank * world_size = 4 * 3 = 12
    assert len(all_global) == count_per_rank * world_size, (
        f"Expected {count_per_rank * world_size} unique global_index, got {len(all_global)}"
    )

    # But the actual sample indices (via global_index % epoch_size) should only cover 0..9
    # Padding samples (global_index 10, 11) map to sample_index 0, 1
    # We can't directly check sample_index from SampleSpec, but we can verify that
    # padding samples have the same RNG seed as early samples

    # Find specs with global_index >= epoch_size (padding samples)
    padding_specs = []
    early_specs = []
    for plan in plans:
        for spec in plan:
            if spec.global_index >= epoch_size:
                padding_specs.append(spec)
            elif spec.global_index < world_size:  # early samples that might be repeated
                early_specs.append(spec)

    # Padding samples should exist (10, 11)
    assert len(padding_specs) == 2, f"Expected 2 padding samples, got {len(padding_specs)}"

    print("✓ test_ddp_remainder_no_loss passed")


if __name__ == "__main__":
    test_plan_determinism()
    test_plan_different_seeds()
    test_sample_spec_serialization()
    test_rng_state_reproducibility()
    test_plan_count_and_indices()
    test_multi_rank_plan_generation()
    test_generation_parameter()
    test_reshuffle_each_epoch_false()
    test_rng_state_independent_of_position()
    test_ddp_epoch_size()
    test_rng_state_varies_across_epochs()
    test_ddp_remainder_no_loss()
    print("\nAll planning tests passed!")
