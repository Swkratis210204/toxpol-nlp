# toxpol-nlp

NLP toolkit for **toxicity and polarization research**. Provides tools for synthetic dataset generation and polarization detection in human annotation studies.

## Install

```bash
pip install toxpol-nlp
```

## Repository Structure

```
data_gen/          synthetic annotation dataset generator
polarized_trees/   Polarized Trees detection algorithm
toxpol/            installable package (source code)
```

### `data_gen/`
Tools for generating synthetic annotation datasets with **injected, known polarization**. Real annotation data cannot provide ground truth for which demographic dimensions drive disagreement — this module does. The generated datasets are the primary validation input for the Polarized Trees algorithm.

→ See [`data_gen/README.md`](data_gen/README.md) for the full API and usage.

### `polarized_trees/`
The Polarized Trees detection algorithm. Given an annotation dataset, it identifies which demographic dimensions split annotators into opposing rating poles and at what severity.

→ Coming soon.

## Tools

| Module | Description | Status |
|---|---|---|
| `toxpol.datagen` | Synthetic annotator pool with injected, ground-truth polarization | Stable |
| `toxpol.trees` | Polarized Trees detection algorithm | Coming soon |

