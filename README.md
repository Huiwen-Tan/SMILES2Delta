# SMILES2Delta

An interactive machine learning pipeline for predicting molecular energy corrections (ΔE) from SMILES using Morgan fingerprints and classical regression models.

## Features

- Morgan fingerprint generation from SMILES (RDKit)
- Models: Linear / Ridge / Random Forest / SVR
- Train / validation / test evaluation (MAE, RMSE, R²)
- Export of test-set predictions and parity plots

## Installation

Clone the repository:

```bash
git clone https://github.com/Huiwen-Tan/SMILES2Delta.git
cd SMILES2Delta
```

Create and activate the Conda environment:

```bash
make env
conda activate s2delta
```

## Usage

Prepare a CSV with at least a SMILES column and a ΔE column. Then run:

```bash
python train.py
```

The script interactively asks for column indices, fingerprint settings, data split,
model choice / hyperparameters, and an output directory for predictions and plots.

## Workflow

1. Load SMILES and reference ΔE values from a CSV.
2. Convert each SMILES into a Morgan fingerprint.
3. Split into train / validation / test sets.
4. Fit a regression model; tune on validation, then evaluate once on test.
5. Save test predictions and (optionally) a parity plot.

## Citation

If you use SMILES2Delta in your research, please cite:

```bibtex
@software{tan_smiles2delta_2026,
  author  = {Tan, Huiwen},
  title   = {{SMILES2Delta}: Machine Learning for Molecular Energy Corrections},
  year    = {2026},
  url     = {https://github.com/Huiwen-Tan/SMILES2Delta},
  license = {MIT}
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
