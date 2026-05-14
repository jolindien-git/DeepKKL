# KKL – Flow Matching

Code accompanying the paper *"Generative Nonlinear Observer for Indistinguishable Systems."*

## Evaluation with Pre-trained Models

To evaluate the pre-trained models, run:
```bash
python scripts/main_cfm.py --epoch 0 --dataset [MODEL] --noise_std [NOISE] --name [NAME]
```
where:
* `[MODEL]` $\in$ {`VDP`, `BiModal`, `VDP2`, `Duffing`}
* `[NOISE]` and `[NAME]` combinations can be chosen as:
* `0.` and `noiseless`
* `1.` and `noisy`

**Example:**

```bash
python scripts/main_cfm.py --epoch 0 --dataset BiModal --noise_std 1. --name noisy

```

## Training from Scratch

To retrain the models, simply remove the `--epoch` argument:

```bash
python scripts/main_cfm.py --dataset BiModal --noise_std 1. --name noisy

```

For `VDP2` (Coupled Van der Pol), the latent dimension should be increased due to a higher state-space dimension. This is done using the `--z_dim` option:

```bash
python scripts/main_cfm.py --dataset VDP2 --noise_std 1. --name noisy --z_dim 12

```

## Baselines

To run all baselines and reproduce the results from **Table I** of the paper, use:

```bash
python scripts/evaluate_all.py --dataset [MODEL] --noise_std [NOISE]

```
