# `toxpol.datagen` — Synthetic Annotation Dataset Generator

Builds a pool of annotators with explicit demographic identities and generates structured disagreement patterns where rating behavior is governed by per-dimension bias configurations.

Because the bias configuration is returned alongside the dataset, recovery can be tested directly against ground truth: did the algorithm find the right dimensions and the right poles?

## Install

```bash
pip install toxpol-nlp

# with nDFU support (for analysis methods)
pip install "toxpol-nlp[ndfu]"
```

## Quickstart

```python
from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS

pool = AnnotatorPool(DEFAULT_DIMENSIONS)
pool.summary()
# Active dimensions : ['gender', 'politics', 'age', 'education', 'orientation']
# Pool size         : 1620  (162 identities × 10 annotators each)

dataset, bias_config = pool.generate_dataset(
    n_texts=100,
    n_annotators_per_text=100,  # must be ≤ pool.pool_size
    noise=0.1,                  # prob. of random rating (outlier noise)
    polarizing_prob=0.7,        # prob. each dimension is polarizing vs. unimodal
)
# dataset columns: text_id, annotator_id, <dimensions>, rating
# bias_config: ground-truth role and poles for each dimension
```

## How it works

Each demographic dimension is independently assigned a role:

- **Polarizing** — splits annotators into opposing rating poles (toxic vs. civil), simulating genuine demographic-driven disagreement
- **Unimodal** — converges all annotators toward one rating range, simulating consensus

`polarizing_prob` controls how many dimensions are polarizing on average. `noise` injects random ratings to simulate real-world outlier disagreement.

## API

### `AnnotatorPool(dimensions, ...)`

| Parameter | Default | Description |
|---|---|---|
| `dimensions` | required | `dict[str, list[str]]` — demographic axes and their values |
| `exclude` | `None` | dimension names to drop (for ablations) |
| `annotators_per_identity` | `10` | replication factor per unique identity combination |
| `scale` | `5` | max rating value; ratings are integers in `[1, scale]` |
| `toxic_range` | `(4, 5)` | rating range for toxic-pole annotators |
| `civil_range` | `(1, 2)` | rating range for civil-pole annotators |
| `neutral_range` | `(3, 3)` | rating range when a unimodal dimension converges to neutral |

### `pool.generate_dataset(n_texts, n_annotators_per_text, noise, polarizing_prob)`

Returns `(dataset, bias_config)`. All texts in one call share the same bias config — call again for a fresh one.

### Diagnostics

```python
pool.pool_size      # int — total annotators
pool.n_identities   # int — unique demographic combinations
pool.active_dims    # dict — dimensions after applying exclude
pool.summary()      # prints full pool config

pool.describe_bias(bias_config)
# dimension    role         details
# ------------------------------------------------------------
# gender       unimodal     convergence=toxic
# politics     polarizing   toxic=['left', 'right']  civil=['center']

pool.summarize(dataset, bias_config, text_id=0)
# Text 0 — overall nDFU: 0.262
# politics (polarizing):
#   center: 0.667
#   left:   0.176

results = pool.analyze(dataset, bias_config)  # raw nDFU scores
# results[text_id]["overall"]      -> float
# results[text_id][dim][value]     -> float

pool.summarize_all(dataset, bias_config)
# Overall nDFU — mean: 0.721  min: 0.431  max: 0.963  (across 100 texts)
```

`analyze()`, `summarize()`, and `summarize_all()` require `pip install "toxpol-nlp[ndfu]"`.

## Default Dimensions

```python
DEFAULT_DIMENSIONS = {
    "gender":      ["male", "female", "non-binary"],
    "politics":    ["left", "center", "right"],
    "age":         ["<25", "25-50", ">50"],
    "education":   ["low", "medium", "high"],
    "orientation": ["heterosexual", "lgbtq+"],
}
# → 162 unique identities → 1620 annotators (default annotators_per_identity=10)
```

Pass any custom dict to use different axes or values:

```python
pool = AnnotatorPool({
    "politics": ["left", "center", "right"],
    "age":      ["<25", ">25"],
})
# 6 identities → 60 annotators
```

## Demo

See [`datagen_demo.ipynb`](datagen_demo.ipynb) for an end-to-end walkthrough with visualizations.
