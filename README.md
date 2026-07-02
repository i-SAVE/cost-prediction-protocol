# When Does Model Complexity Pay Off? — Reproducibility Package

Code and results for the manuscript:

> Isaev, S. *When Does Model Complexity Pay Off? A Leakage-Controlled Statistical Protocol for Cost Prediction with Tabular Models and Graph Neural Networks.* Submitted to the Asian Journal of Civil Engineering, 2026.

Every table and figure in the paper can be regenerated from this repository on a CPU-only machine.

## Contents

```
scripts/
  run_block_a.py            # Block A: 6-model tabular benchmark on Ames Housing
                            #   (leakage-controlled pipeline, randomised search,
                            #    bootstrap CIs, Wilcoxon, TreeSHAP, OOF error model)
  block_b_gatv2_ablation.py # Block B: GATv2 vs MLP ablation on the controlled
                            #   synthetic WBS graph (JAX; 10 repeated splits)
results/                    # exported result files used in the paper
figures/                    # 300-dpi figures as they appear in the paper
requirements.txt            # pinned library versions
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd scripts
python run_block_a.py              # ~5 min CPU
python block_b_gatv2_ablation.py   # ~1 min CPU
```

Outputs are written to `scripts/results_rerun/` and should match `results/` up to
floating-point noise (all seeds are fixed).

## Data

* **Block A** — Ames Housing (De Cock, 2011; N = 2930). Loaded from the `rdatasets`
  Python package (`openintro/ames`). Alternative source: OpenML dataset 42165
  (`sklearn.datasets.fetch_openml(data_id=42165)`).
* **Block B** — synthetic 300-node work-breakdown-structure DAG generated inside the
  script with fixed seeds (graph: 42; target noise: 49; splits: 1000–1009). The
  generative process (including the 58% upstream variance share) is documented in the
  script and in Section 3.7 of the paper.

## Fixed seeds

| Component | Seed |
|---|---|
| Global / splits / models (Block A) | 42 |
| Bootstrap resampling | 123 |
| Graph generator | 42 (+7 for target noise) |
| Repeated node splits (Block B) | 1000–1009 |

## Environment used for the paper

Python 3.10 · scikit-learn 1.7.2 · XGBoost 3.2.0 · JAX 0.6.2 · NumPy 2.2.6 ·
SciPy 1.15.3 · pandas 2.3.3 · CPU-only Linux; no GPU required.

## Note on the GATv2 implementation

The GATv2 layer is implemented in JAX following the official PyTorch-Geometric
`GATv2Conv` semantics (separate source/target projections, attention applied after
the LeakyReLU non-linearity, self-loops added). See Section 3.8 of the paper.

## License

MIT (see `LICENSE`).
