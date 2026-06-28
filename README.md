# toxpol-nlp

NLP toolkit for **toxicity and polarization research**. Provides tools for synthetic dataset generation and polarization detection in human annotation studies.

Built around **Polarized Trees** — a framework that explains *why* annotators disagree on a text by finding demographic splits, rather than collapsing disagreement into a binary verdict.

## Install

```bash
pip install toxpol-nlp
```

## Tools

| Module | Description | Docs | Status |
|---|---|---|---|
| `toxpol.datagen` | Synthetic annotator pool with injected, ground-truth polarization | [docs/datagen.md](docs/datagen.md) | Stable |
| `toxpol.trees` | Polarized Trees detection algorithm | — | Coming soon |

## Quickstart

```python
from toxpol.datagen import AnnotatorPool, DEFAULT_DIMENSIONS

pool = AnnotatorPool(DEFAULT_DIMENSIONS)
dataset, bias_config = pool.generate_dataset(n_texts=100, n_annotators_per_text=100)
```

See each tool's documentation for the full API and examples.
