"""
Synthetic annotation dataset generator for studying demographic polarization in human labeling.

Builds a pool of annotators with explicit demographic identities and generates structured
disagreement patterns where rating behavior is governed by per-dimension bias configurations.
"""

import itertools
import random

import numpy as np
import pandas as pd


# Default demographic dimensions used in the paper
DEFAULT_DIMENSIONS = {
    "gender": ["male", "female", "non-binary"],
    "politics": ["left", "center", "right"],
    "age": ["<25", "25-50", ">50"],
    "education": ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}


class AnnotatorPool:
    """
    Synthetic annotator pool for generating polarized rating datasets.

    Builds a Cartesian-product pool of demographic identities, then generates
    annotation datasets where each dimension is randomly assigned either a
    "polarizing" role (splitting annotators into toxic/civil poles) or a
    "unimodal" role (converging all annotators toward one rating range).

    Parameters
    ----------
    dimensions : dict[str, list[str]]
        Mapping from dimension name to the list of possible values.
        Example: {"politics": ["left", "center", "right"], "age": ["<25", ">25"]}

    exclude : list[str] | None
        Dimension names to drop before building identities. Useful for ablations.

    annotators_per_identity : int
        How many annotators share each unique demographic combination.
        Pool size = product(len(v) for v in dimensions.values()) * annotators_per_identity.

    scale : int
        Maximum value on the rating scale (ratings are integers in [1, scale]).

    toxic_range : tuple[int, int]
        (low, high) inclusive range from which toxic-pole annotators draw ratings.

    civil_range : tuple[int, int]
        (low, high) inclusive range from which civil-pole annotators draw ratings.

    neutral_range : tuple[int, int]
        (low, high) inclusive range used when a unimodal dimension converges to "neutral".

    Examples
    --------
    >>> from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS
    >>> pool = AnnotatorPool(DEFAULT_DIMENSIONS)
    >>> dataset, bias_config = pool.generate_dataset(n_texts=50, n_annotators_per_text=100)
    >>> dataset.head()
    """

    def __init__(
        self,
        dimensions,
        exclude=None,
        annotators_per_identity=10,
        scale=5,
        toxic_range=(4, 5),
        civil_range=(1, 2),
        neutral_range=(3, 3),
    ):
        self.annotators_per_identity = annotators_per_identity
        self.identities, self.active_dims = self._get_identities(dimensions, exclude)
        self.pool = self._build_pool()

        self.scale = scale
        self.toxic_range = toxic_range
        self.civil_range = civil_range
        self.neutral_range = neutral_range

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _get_identities(self, dimensions, exclude=None):
        active_dims = {k: v for k, v in dimensions.items() if k not in (exclude or [])}
        identities = [
            dict(zip(active_dims.keys(), combo))
            for combo in itertools.product(*active_dims.values())
        ]
        return identities, active_dims

    def _build_pool(self):
        pool = []
        for identity in self.identities:
            for _ in range(self.annotators_per_identity):
                pool.append(identity.copy())
        pool = pd.DataFrame(pool)
        pool.index.name = "annotator_id"
        return pool

    # ------------------------------------------------------------------
    # Bias configuration
    # ------------------------------------------------------------------

    def _generate_bias_config(self, polarizing_prob=0.7):
        """
        Randomly assign each active dimension a role for one dataset instance.

        A "polarizing" dimension splits its values into a toxic pole and a civil
        pole. An "unimodal" dimension converges all annotators toward one range.

        Returns
        -------
        dict
            Keys are dimension names. Each value is a dict with:
            - role: "polarizing" | "unimodal"
            - toxic_pole / civil_pole (if polarizing): lists of dimension values
            - convergence (if unimodal): "toxic" | "civil" | "neutral"
        """
        config = {}
        for dim, values in self.active_dims.items():
            role = random.choices(
                ["polarizing", "unimodal"],
                weights=[polarizing_prob, 1 - polarizing_prob],
            )[0]
            if role == "polarizing":
                shuffled = values.copy()
                random.shuffle(shuffled)
                split = random.randint(1, len(shuffled) - 1)
                config[dim] = {
                    "role": "polarizing",
                    "toxic_pole": shuffled[:split],
                    "civil_pole": shuffled[split:],
                }
            else:
                config[dim] = {
                    "role": "unimodal",
                    "convergence": random.choice(["toxic", "civil", "neutral"]),
                }
        return config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_texts=100,
        n_annotators_per_text=100,
        noise=0.1,
        polarizing_prob=0.7,
    ):
        """
        Generate a synthetic annotation dataset.

        A single bias configuration is drawn for the entire dataset (all texts
        share the same demographic polarization structure). Each text is then
        annotated by a random subset of the pool.

        Parameters
        ----------
        n_texts : int
            Number of texts to annotate. Must be >= 1.

        n_annotators_per_text : int
            Annotators sampled per text (without replacement).
            Must be <= pool size (annotators_per_identity * number_of_identities).

        noise : float in [0, 1]
            Probability that any annotator ignores the bias config and draws
            a uniformly random rating instead.

        polarizing_prob : float in [0, 1]
            Prior probability that each dimension is assigned a "polarizing"
            role in the bias config (vs. "unimodal").

        Returns
        -------
        dataset : pd.DataFrame
            One row per (text_id, annotator_id) pair. Columns:
            text_id, annotator_id, <all active dimension columns>, rating.

        bias_config : dict
            The bias configuration used for this dataset. See
            `_generate_bias_config` for the structure.
        """
        if n_annotators_per_text > len(self.pool):
            raise ValueError(
                f"n_annotators_per_text ({n_annotators_per_text}) exceeds pool size "
                f"({len(self.pool)}). Reduce n_annotators_per_text or increase "
                f"annotators_per_identity."
            )

        bias_config = self._generate_bias_config(polarizing_prob)

        # Precompute vote lookup for each polarizing dimension:
        # maps annotator value → "toxic" | "civil"
        vote_maps = {}
        unimodal_fallback = None
        for dim, config in bias_config.items():
            if config["role"] == "polarizing":
                vote_maps[dim] = (
                    {v: "toxic" for v in config["toxic_pole"]}
                    | {v: "civil" for v in config["civil_pole"]}
                )
            elif unimodal_fallback is None:
                unimodal_fallback = config["convergence"]

        frames = []
        for text_id in range(n_texts):
            sampled = self.pool.sample(n=n_annotators_per_text, replace=False)

            if vote_maps:
                # Vectorised majority vote across all polarizing dimensions
                toxic = np.zeros(len(sampled), dtype=np.int32)
                civil = np.zeros(len(sampled), dtype=np.int32)
                for dim, vmap in vote_maps.items():
                    mapped = sampled[dim].map(vmap)
                    toxic += (mapped == "toxic").values
                    civil += (mapped == "civil").values
                label = np.where(toxic > civil, 0,
                         np.where(civil > toxic, 1, 2))  # 0=toxic,1=civil,2=neutral
            else:
                fb = {"toxic": 0, "civil": 1, "neutral": 2}[unimodal_fallback or "neutral"]
                label = np.full(len(sampled), fb, dtype=np.int32)

            # Draw ratings from the appropriate range per label
            ranges = [self.toxic_range, self.civil_range, self.neutral_range]
            ratings = np.array([
                np.random.randint(ranges[l][0], ranges[l][1] + 1) for l in label
            ])

            # Inject noise
            noise_mask = np.random.random(len(sampled)) < noise
            ratings[noise_mask] = np.random.randint(1, self.scale + 1, noise_mask.sum())

            frame = sampled.copy()
            frame.insert(0, "text_id", text_id)
            frame["rating"] = ratings
            frames.append(frame)

        dataset = pd.concat(frames)
        dataset.index.name = "annotator_id"
        return dataset, bias_config

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    @property
    def pool_size(self):
        """Total number of annotators in the pool."""
        return len(self.pool)

    @property
    def n_identities(self):
        """Number of unique demographic identity combinations."""
        return len(self.identities)

    def summary(self):
        """Print a brief summary of the pool configuration."""
        print(f"Active dimensions : {list(self.active_dims.keys())}")
        print(f"Unique identities : {self.n_identities}")
        print(f"Annotators/identity: {self.annotators_per_identity}")
        print(f"Pool size          : {self.pool_size}")
        print(f"Rating scale       : 1–{self.scale}")
        print(f"  toxic_range      : {self.toxic_range}")
        print(f"  civil_range      : {self.civil_range}")
        print(f"  neutral_range    : {self.neutral_range}")

    def describe_bias(self, bias_config):
        """Pretty-print the bias config as a readable table."""
        col_w = max(len(d) for d in bias_config) + 2
        print(f"{'dimension':<{col_w}} {'role':<12} details")
        print("-" * 60)
        for dim, config in bias_config.items():
            if config["role"] == "polarizing":
                details = (
                    f"toxic={config['toxic_pole']}  civil={config['civil_pole']}"
                )
            else:
                details = f"convergence={config['convergence']}"
            print(f"{dim:<{col_w}} {config['role']:<12} {details}")

    def analyze(self, dataset, bias_config):
        """
        Compute nDFU scores for every text, overall and per dimension value.

        Requires the `ndfu` package (`pip install toxpol-nlp[ndfu]`).

        Returns
        -------
        dict
            results[text_id]["overall"] -> float
            results[text_id][dim][value] -> float
        """
        try:
            from ndfu import dfu
        except ImportError:
            raise ImportError(
                "ndfu is required for analyze(). "
                "Install it with: pip install toxpol-nlp[ndfu]"
            )

        def _ndfu(ratings):
            counts = np.bincount(ratings, minlength=self.scale + 1)[1:]
            hist = counts / counts.sum()
            return dfu(hist)

        results = {}
        for text_id, text_data in dataset.groupby("text_id"):
            text_results = {"overall": _ndfu(text_data["rating"].values)}
            for dim in self.active_dims:
                text_results[dim] = {
                    value: _ndfu(group["rating"].values)
                    for value, group in text_data.groupby(dim)
                }
            results[text_id] = text_results
        return results

    def summarize(self, dataset, bias_config, text_id=0):
        """
        Print nDFU scores for one text, grouped by dimension, with bias roles shown.

        Calls analyze() internally. Requires `ndfu`.
        """
        results = self.analyze(dataset, bias_config)
        text = results[text_id]
        print(f"Text {text_id} — overall nDFU: {text['overall']:.3f}\n")
        for dim, values in text.items():
            if dim == "overall":
                continue
            role = bias_config[dim]["role"]
            print(f"{dim} ({role}):")
            for value, score in values.items():
                print(f"  {value}: {score:.3f}")
            print()

    def summarize_all(self, dataset, bias_config):
        """
        Print mean nDFU per dimension value, aggregated across all texts.

        Gives a compact cross-text view: instead of per-text scores, each
        dimension value shows its average nDFU and the spread (min–max).
        Useful for seeing which demographic groups consistently disagree
        more across the whole dataset.

        Calls analyze() internally. Requires `ndfu`.
        """
        results = self.analyze(dataset, bias_config)
        n_texts = len(results)

        overall_scores = [results[t]["overall"] for t in results]
        print(f"Overall nDFU — mean: {np.mean(overall_scores):.3f}  "
              f"min: {np.min(overall_scores):.3f}  "
              f"max: {np.max(overall_scores):.3f}  "
              f"(across {n_texts} texts)\n")

        for dim in self.active_dims:
            role = bias_config[dim]["role"]
            print(f"{dim} ({role}):")
            for value in self.active_dims[dim]:
                scores = [results[t][dim][value] for t in results if value in results[t][dim]]
                print(f"  {value:<15} mean: {np.mean(scores):.3f}  "
                      f"min: {np.min(scores):.3f}  "
                      f"max: {np.max(scores):.3f}")
            print()
