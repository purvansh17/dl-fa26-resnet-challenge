# Efficient ResNet for CIFAR-10

**Purvansh Arora · Ricardo Diaz — Team Aegis**  
NYU Tandon Deep Learning, Fall 2026

A residual CNN trained from scratch on course-supplied images under a **5 million trainable parameter** limit. Our final system uses a **4,298,970-parameter WRN-22-4** student trained with spatial attention transfer from a larger, frozen SE-ResNet teacher. The teacher is discarded at inference.

## Results

Final standing: **35th of 81 teams**, with **87.2% private accuracy** in the [competition](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).

| Selected submission | Attention coefficient | Public accuracy | Private accuracy |
|---|---:|---:|---:|
| Original strength | 14.22635 | **89.5%** | 86.4% |
| Double strength | 28.45271 | 89.3% | **87.2%** |

Both entries were selected before private results were revealed. Each used one student network, with original/flip mean logits and frozen BatchNorm. Their predictions were not ensembled.

## Method

- **Student:** three residual groups with widths 64, 128 and 256; three blocks per group; global average pooling and a ten-class head.
- **Teacher:** 15,904,586-parameter SE-ResNet, trained from scratch on the same permitted 45k training split; used only during training.
- **Attention transfer:** match channel-averaged, L2-normalized spatial energy maps at three stages. The objective adds the weighted attention loss to classification loss.
- **Training:** 200 epochs, batch 256, SGD with momentum, OneCycle maximum learning rate 0.1, tensor CIFAR AutoAugment, CutMix and label smoothing.
- **Development:** fixed 45k/5k split using pandas shuffle seed 42. Repeated use makes this development evidence, not an independent test set.

No external training data or externally pretrained weights were used for the final models. See the [preserved method/results record](docs/reports/ATTENTION_POST_SUBMISSION.md) for detailed comparisons and limitations.

## Repository

| Path | Contents |
|---|---|
| `src/` | Final model, attention loss, augmentation, data loading, training and prediction |
| `configs/` | Original/double-strength recipes and source provenance |
| `kaggle_runs/` | Preserved sources of executed experiments |
| `notebooks/` | Teacher training and selected historical team baseline notebooks |
| `notebooks/DL_resnet_ricardo.ipynb`, `notebooks/ricardo_experiments/` | Four-stage ResNet branch: notebook, ConvNeXt, degradation augmentation, attention-transfer distillation and the W&B plotting script for the report figures |
| `docs/reports/` | Experiment results and verification summaries |

`src/train.py` is the final attention-student trainer. `src/train_resnet.py` is an earlier SE-ResNet implementation, retained for historical experiments.

## Setup

Run commands from the repository root. A Python 3.12 environment was used for local verification. Install a compatible PyTorch/torchvision pair for your platform; full training requires CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Requirements are not an exact environment lock. Historical T4 training used PyTorch 2.11.0+cu130; local CPU parity checks used PyTorch 2.14.1 and torchvision 0.29.1.

## Reproduction

Course data and trained checkpoints are excluded from Git and must be supplied locally. Exact historical-checkpoint verification is therefore not available from a clone alone.

Arrange the course files as follows:

```text
data/
  train_labels.csv       # id,filename,label
  train_images/
  test.csv               # id,filename
  test_images/
  sample_submission.csv  # id,label
```

### Train the student

The supplied configs verify the SHA-256 of the team's original teacher checkpoint. Teacher training code is preserved in the [teacher-training notebook](notebooks/large_teacher_distillation_colab%20%281%29.ipynb); its original environment paths require adjustment. A newly trained teacher needs a separately documented config with its own checkpoint hash.

```bash
python -m src.train --course-root data \
  --teacher models/teacher_best.pth \
  --config configs/final_double.json \
  --output data/reproduction_double
```

Use `configs/final_original.json` for original attention strength. Add `--resume` to restore a run created by this trainer, including optimizer, scheduler, scaler and epoch. Historical workers use a different recovery format.

### Generate predictions

With the original double-strength checkpoint available locally:

```bash
python -m src.predict --course-root data \
  --checkpoint models/candidates/attention_beta_double_ep195/best_flip.pth \
  --checkpoint-sha256 26f3295e5d03b71b69b9ae4d874c7232713e68f710cbce1f7fe13f392d4994ab \
  --device cpu --output data/reproduction_export/submission.csv
```

Use `--device cuda` for GPU inference. The command checks checkpoint identity and official image-ID order, writes `id,label` plus a provenance manifest, and never uploads to Kaggle. Hidden labels are unavailable; final accuracy comes from the official leaderboard.

### Verify the extracted implementation

```bash
python -m src.verify
```

This requires course images and the original team artifacts at the paths specified in the script. Both selected checkpoints matched historical-source CPU FP32 logits exactly on 16 development images. Sampled augmentation, classification/attention losses and student gradients also matched. See the [verification summary](docs/reports/READABLE_PIPELINE_VERIFICATION.json).

These checks establish bounded implementation parity. A fresh end-to-end CUDA run of the extracted trainer has not been completed, and cross-version bitwise reproduction is not guaranteed.

## Acknowledgements

The method builds on [Wide Residual Networks](https://arxiv.org/abs/1605.07146), [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), [Attention Transfer](https://arxiv.org/abs/1612.03928), [AutoAugment](https://arxiv.org/abs/1805.09501) and [CutMix](https://arxiv.org/abs/1905.04899). The attention-map definition follows the authors' [`at` and `at_loss` implementation](https://github.com/szagoruyko/attention-transfer/blob/master/utils.py). Selected team notebooks and [executed-source provenance](configs/source_provenance.json) are preserved in this repository.

Gemini, ChatGPT and Codex assisted with code development, experiment planning and orchestration, analysis and documentation. Reported measurements come from saved experiment artifacts and official competition results. Historical notebooks include additional experiments that are not all mapped to verified submission results.
