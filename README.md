# ead_bayesian_dlnm

Forecasting avoidable deaths with various bayesian models.

## Installation

Make the conda environment,

```bash
conda env create -f EAD_BAYESIAN_DLNM_env.yml
```

To update the conda environment (after adding new packages) use,

```bash
conda env update -f EAD_BAYESIAN_DLNM_env.yml --prune
```

## Feature engineering

```bash
cd scripts
Rscript data_cleaning.R
```

## DLNM model

```bash
cd scripts
python build_model.py
```