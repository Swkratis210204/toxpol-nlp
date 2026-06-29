import pytest
from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS

SMALL_DIMS = {
    "politics": ["left", "center", "right"],
    "age": ["<25", ">25"],
}

# Shared default ranges for SMALL_DIMS tests (scale=5)
SCALE = 5
TOXIC_RANGE = (4, 5)
CIVIL_RANGE = (1, 2)
NEUTRAL_RANGE = (3, 3)


def make_pool(**overrides):
    """Helper: build an AnnotatorPool on SMALL_DIMS with sensible scale=5 defaults."""
    kwargs = dict(
        dimensions=SMALL_DIMS,
        scale=SCALE,
        toxic_range=TOXIC_RANGE,
        civil_range=CIVIL_RANGE,
        neutral_range=NEUTRAL_RANGE,
    )
    kwargs.update(overrides)
    return AnnotatorPool(**kwargs)


def test_pool_size():
    pool = make_pool(annotators_per_identity=5)
    assert pool.pool_size == 6 * 5
    assert pool.n_identities == 6


def test_exclude_reduces_dimensions():
    pool = make_pool(exclude=["age"])
    assert "age" not in pool.active_dims
    assert "politics" in pool.active_dims
    assert pool.n_identities == 3


def test_generate_dataset_shape():
    pool = make_pool(annotators_per_identity=10)
    dataset, _ = pool.generate_dataset(n_texts=5, n_annotators_per_text=10)
    assert len(dataset) == 5 * 10


def test_dataset_columns():
    pool = make_pool()
    dataset, _ = pool.generate_dataset(n_texts=2, n_annotators_per_text=5)
    assert "text_id" in dataset.columns
    assert "rating" in dataset.columns
    for dim in SMALL_DIMS:
        assert dim in dataset.columns


def test_ratings_within_scale():
    pool = make_pool(scale=5, toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3))
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10)
    assert dataset["rating"].between(1, 5).all()


def test_ratings_within_scale_custom():
    pool = make_pool(scale=7, toxic_range=(6, 7), civil_range=(1, 2), neutral_range=(3, 4))
    dataset, _ = pool.generate_dataset(n_texts=5, n_annotators_per_text=10)
    assert dataset["rating"].between(1, 7).all()


def test_n_annotators_exceeds_pool_raises():
    pool = make_pool(annotators_per_identity=5)
    with pytest.raises(ValueError, match="exceeds pool size"):
        pool.generate_dataset(n_texts=1, n_annotators_per_text=pool.pool_size + 1)


def test_invalid_ratio_sum_raises():
    pool = make_pool()
    with pytest.raises(AssertionError, match="sum to 1.0"):
        pool.generate_dataset(
            n_texts=2, n_annotators_per_text=5,
            high_ratio=0.5, moderate_ratio=0.3, low_ratio=0.3,
        )


def test_bias_configs_keys_match_text_ids():
    pool = make_pool()
    n = 5
    dataset, bias_configs = pool.generate_dataset(n_texts=n, n_annotators_per_text=5)
    assert set(bias_configs.keys()) == set(range(n))


def test_bias_config_tiers_are_valid():
    pool = make_pool()
    _, bias_configs = pool.generate_dataset(n_texts=10, n_annotators_per_text=5)
    for text_id, cfg in bias_configs.items():
        assert cfg["tier"] in ("high", "moderate", "low")
        if cfg["tier"] in ("high", "moderate") or cfg.get("subcase") == "weighted":
            assert cfg["config"] is not None
            assert "threshold" in cfg
            assert set(cfg["config"].keys()) == set(pool.active_dims.keys())
        else:
            assert cfg["config"] is None
            assert "peak" in cfg
            assert "spread" in cfg


def test_noise_zero_ratings_in_range():
    pool = make_pool(scale=5, toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3))
    dataset, _ = pool.generate_dataset(n_texts=10, n_annotators_per_text=10, noise=0.0)
    assert dataset["rating"].between(1, 5).all()


def test_text_ids_are_correct():
    pool = make_pool()
    n = 7
    dataset, _ = pool.generate_dataset(n_texts=n, n_annotators_per_text=5)
    assert set(dataset["text_id"].unique()) == set(range(n))


def test_default_dimensions():
    pool = AnnotatorPool(
        DEFAULT_DIMENSIONS, scale=5,
        toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3),
    )
    assert pool.n_identities == 162
    assert pool.pool_size == 1620


def test_moderate_and_low_ranges_derived_at_scale_5():
    """At scale=5 with the paper's default ranges, Moderate/Low ranges
    should reproduce the original hand-tuned values exactly."""
    pool = AnnotatorPool(
        DEFAULT_DIMENSIONS, scale=5,
        toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3),
    )
    assert pool.moderate_toxic_range == (4, 5)
    assert pool.moderate_civil_range == (1, 3)
    assert pool.low_toxic_range == (3, 5)
    assert pool.low_civil_range == (1, 3)


def test_moderate_and_low_ranges_scale_with_custom_scale():
    """Moderate/Low ranges should be derived proportionally for a
    different scale, not stuck at scale=5 numbers."""
    pool = AnnotatorPool(
        DEFAULT_DIMENSIONS, scale=10,
        toxic_range=(8, 10), civil_range=(1, 3), neutral_range=(4, 7),
    )
    # step = max(1, round(10/5)) = 2
    assert pool.moderate_toxic_range == (8, 10)
    assert pool.moderate_civil_range == (1, 5)
    assert pool.low_toxic_range == (6, 10)
    assert pool.low_civil_range == (1, 5)


def test_mandatory_args_enforced():
    """scale/toxic_range/civil_range/neutral_range have no defaults."""
    with pytest.raises(TypeError):
        AnnotatorPool(DEFAULT_DIMENSIONS)


def test_tier_ratios_respected_approximately():
    pool = AnnotatorPool(
        DEFAULT_DIMENSIONS, scale=5,
        toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3),
    )
    n = 200
    _, bias_configs = pool.generate_dataset(
        n_texts=n, n_annotators_per_text=50,
        high_ratio=0.6, moderate_ratio=0.2, low_ratio=0.2,
    )
    tiers = [cfg["tier"] for cfg in bias_configs.values()]
    high_share = tiers.count("high") / n
    assert 0.45 < high_share < 0.75  # generous tolerance for a stochastic draw


def test_severity_ordering_high_gt_moderate_gt_low():
    pytest.importorskip("ndfu")
    pool = AnnotatorPool(
        DEFAULT_DIMENSIONS, scale=5,
        toxic_range=(4, 5), civil_range=(1, 2), neutral_range=(3, 3),
    )
    dataset, bias_configs = pool.generate_dataset(
        n_texts=150, n_annotators_per_text=100,
        high_ratio=0.6, moderate_ratio=0.2, low_ratio=0.2,
    )
    results = pool.analyze(dataset, bias_configs)
    tier_scores = {"high": [], "moderate": [], "low": []}
    for text_id, cfg in bias_configs.items():
        tier_scores[cfg["tier"]].append(results[text_id]["overall"])
    import numpy as np
    means = {t: np.mean(s) for t, s in tier_scores.items() if s}
    assert means["high"] > means["moderate"] > means["low"]