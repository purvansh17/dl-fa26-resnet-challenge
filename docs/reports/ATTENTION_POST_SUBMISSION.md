# Attention transfer: method and recorded results

## Verified final outcomes

The competition is complete. Aegis finished **35th of 81 teams with 87.2% private accuracy**, verified from the saved complete final Kaggle leaderboard. The team selected its two highest-public entries before private scores were revealed:

| Selected model | Submission reference | Selected epoch | Attention beta | Public | Private |
|---|---:|---:|---:|---:|---:|
| Original strength | 58026330 | 194 | 14.226353051298315 | **89.5%** | 86.4% |
| Double strength | 58039715 | 195 | 28.45270610259663 | 89.3% | **87.2%** |

Each entry used one 4,298,970-parameter WRN-22-4; the models were not ensembled. Original strength improved on the earlier classification-only WRN's 88.3% public / 86.0% private accuracy. Its public gain was 1.20 percentage points, or 12 net correct predictions on the reported 1,000-image public subset. Hidden paired correctness is unavailable: this does not mean 12 fixes and zero regressions. Double strength ranked lower publicly but higher privately. Final scores are retrospective evidence, not a new tuning set.

This document retains detailed diagnostics for the **original-strength** run; they should not be attributed to the double-strength run unless explicitly stated. Course images, checkpoints, submitted CSVs and raw analyses are retained locally and excluded from Git. Paths under `data/` and `models/` below identify those local artifacts, not files supplied by a public clone.

| Selected artifact | Checkpoint SHA-256 | Submitted CSV SHA-256 |
|---|---|---|
| Original strength | `518035fc36286bd36901063100bfc4ac3853727364e6d03587ce92b0bbf6bd9e` | `ff6a317f7fb7c3d2c427690f1eef5fd278de806e09f5cf0839acb8a81e1d2883` |
| Double strength | `26f3295e5d03b71b69b9ae4d874c7232713e68f710cbce1f7fe13f392d4994ab` | `e0f33ff4bc1b23849d57f8d6274d0787934c3a87b2daa6c7bee916be23c036ff` |

The original release is preserved locally at `models/releases/attention_public0895/`; the selected double checkpoint is `models/candidates/attention_beta_double_ep195/best_flip.pth`. Earlier releases remain intact. Local final evidence is recorded in `data/final_results_20261109/verified_summary.json`, `submissions.csv` and `leaderboard.csv`.

## Original-strength recipe

| Component | Successful configuration |
|---|---|
| Deployed model | One WRN-22-4, **4,298,970 actual trainable parameters** |
| Origin and data | Random initialization; course-supplied images only |
| Split | Original pandas shuffle seed 42: first 45,000 training / last 5,000 development |
| Split signature | `a805b2cd9b1c8ff89972372ea90dafa82090c090c00eafcdbfc3a479a45d56cb` |
| Training | 200 epochs, seed 42, batch 256, raw weights; no EMA |
| Normalization | Mean (0.4914, 0.4822, 0.4465); std (0.2023, 0.1994, 0.2010) |
| Randomness | Epoch seed `42 + epoch_index*971` for torch/NumPy/Python; cuDNN benchmark enabled, not deterministic CUDA |
| Optimizer | SGD initial momentum 0.9; OneCycle cycles momentum 0.85–0.95; weight decay 0.0005 |
| Schedule | OneCycle max LR 0.1, default `pct_start=0.3`, per batch |
| Augmentation | Shared team GPU tensor CIFAR AutoAugment; zero-pad random crop; horizontal flip |
| Mixing and regularization | CutMix alpha 1, probability 0.5; CE label smoothing 0.1; gradient clip 1 |
| Teacher | Own-data SE teacher, 15,904,586 parameters; frozen weights and BN, no gradients |
| Auxiliary objective | Spatial attention transfer, fixed beta **14.226353051298315** |
| Precision | AMP training on T4; final validation/export FP32 |
| Selection | Highest training-time AMP development flip accuracy, earliest on ties; selected **epoch 194**, then evaluated/exported FP32 |
| Inference | Frozen source BN; average original and horizontal-flip **logits**, then argmax |
| Excluded | No external weights/data, hidden training, adapted BN, balancing, teacher at inference or ensemble |

Teacher checkpoint SHA256: `609fe3d18cda5e36e86025e6e46e17a23f169db017118f7b9b7a6125b88ec565`. Teacher layers 1/2/3 and student groups 1/2/3 align at 32/16/8 pixels. Both receive the **same final augmented/CutMix view**. For each stage, compute channel-mean squared activations, flatten spatially, L2-normalize per image, and take the mean squared difference. Sum the three stage losses, multiply by beta and add smoothed CutMix CE. No trainable projection or logits-KD objective is added. The teacher is discarded after training.

Selected historical training/export sources are retained under `kaggle_runs/attention_candidate/`, `attention_beta_pair/`, `attention_predictions/` and `beta_double_predictions/`. Maintained entry points are `python -m src.train`, `python -m src.predict` and `python -m src.verify`; configs are `configs/final_original.json` and `configs/final_double.json`. Tensor and PIL AutoAugment are not claimed to produce identical sampled views. The supplied configs pin the historical own-data teacher; a newly trained teacher requires a separately documented config and hash.

## What actually changed versus public 0.883

An AST/config audit confirmed **12 identical declarations** covering the WRN architecture, initialization, normalization, augmentation, CutMix, evaluation, epoch seeding and final predictions, plus **14 identical configuration fields** covering split, seed, architecture, training budget, optimizer/schedule and selection. Saved audit: `data/attention_recipe_audit.json`.

The intended treatment change is the frozen teacher's spatial-attention loss, with its additional forward pass/hooks and loss/AMP bookkeeping. This is substantially better controlled than earlier experiments that changed architecture and augmentation together. It remains a **historical single-seed comparison**, not a concurrent deterministic replication. Source parity does not establish an identical CUDA trajectory or exclude run variation. Our evidence supports this particular configuration; it does not prove a universal causal advantage for attention transfer.

## Calibration of the auxiliary loss mattered

The initial beta 1000 pilot was unsuitable as an unchecked default: FP32 gradient measurements on four permitted training batches showed attention/CE backbone norm ratios **25.71–42.77**, median **35.146**. Scalar loss ratios understated this imbalance. A rule fixed before seeing calibration results targeted an initial norm ratio of 0.5: `min(1000, 500 / median_ratio)`, giving beta 14.22635. No validation sweep selected this coefficient.

At selected epoch 194, mean CE was 0.87940 and weighted attention loss 0.01361. Their scalar ratio is not their gradient ratio; initial calibration does not guarantee later balance. Do not increase beta merely to make the displayed losses numerically similar, or copy this coefficient to a different architecture/map reduction without checking scale.

The full run logged 110.9 training minutes on T4. Pilot median step times were 0.122s CE and 0.180s attention: teacher guidance increased training work while leaving inference parameter count and structure unchanged. There were 12 skipped AMP updates out of 35,200 batches; training continued and final states/logits passed finite checks. No inference speed improvement was measured.

## Local gain was small, but not confined to one epoch

Selected FP32 flip validation: **97.30% → 97.46%**, 45 fixes and 37 regressions, net +8/5,000. Paired bootstrap 95% interval: **[−0.18, +0.52] percentage points**; exact McNemar p=0.4397. These are reused-development diagnostics, not independent confirmation or hidden-win probabilities.

New trajectory analysis found:

- Attention exceeded the historical WRN's development flip score in **18 of the final 20 epochs**, losing in two.
- Last-20 median: **97.17% versus 97.00%**; mean same-epoch difference +0.215pp.
- Final epoch 200: **97.36% versus 97.10%**.

This makes a single lucky selected epoch a less complete explanation of the local result. The epochs are correlated measurements on one trajectory and the same development set; do not treat 18/20 as 18 independent wins. Saved analysis: `data/attention_training_trajectory_analysis.json`.

## Class and distribution tradeoffs

| Development observation | Change |
|---|---:|
| Cat accuracy, correct images | +7 (15 fixes / 8 regressions) |
| Dog accuracy, correct images | No net change (8 / 8) |
| Cat→dog errors | 24 → 19 |
| Dog→cat errors | 18 → 18 |
| Combined cat/dog confusion | **42 → 37** |
| Ship errors | **5 → 11**, no fixes / 6 regressions |
| Truck correct images | +5, no regressions |

These are development findings only. They cannot identify which public classes improved.

| Fixed condition | Accuracy change versus 0.883 model |
|---|---:|
| Clean | +0.16pp |
| Brightness | +0.34pp |
| Contrast | −0.04pp |
| Noise std 0.035 | **−1.66pp** |
| Full 3×3 average blur | **+3.30pp** |
| Grayscale | +0.42pp |
| JPEG40 | **+2.28pp** |

Eight additional mild/stress conditions differed only −0.06 to +0.22pp. No weighted aggregate was used. A plausible hypothesis is that feature transfer changed which appearance variations the student handles well. **We cannot conclude that blur, JPEG, cats, or any particular hidden distribution caused the public gain.** Do not inspect hidden images or recover labels to explain it.

Clean NLL/Brier/ECE worsened despite accuracy improving. Both models were underconfident. Probability quality and classification accuracy answer different questions; poor calibration is not an automatic veto for an accuracy competition. Nor does this success make calibration irrelevant in general.

Stability also needs labels: at noise std 0.02 attention had higher clean/transformed agreement (95.60% vs 95.24%) but slightly lower accuracy (94.32% vs 94.36%). Stable errors can inflate agreement. The labeled stability breakdown is the useful diagnostic, not unlabeled hidden agreement.

## What this teaches us about the search

**Supported:** this original-strength student exceeded the earlier classification-only WRN on both public and private aggregate accuracy while retaining the same legal inference architecture. The selected double-strength variant reached the best private score among the selected entries. Large own-data training-only capacity was a productive direction in these recorded runs. The teacher's modest clean advantage did not prevent a useful student; teacher top-1 accuracy alone was an incomplete screening measure for feature transfer.

**Not established:** a statistically reliable or universal advantage, the hidden-set identity, hidden per-class gains, the optimal beta/stages/teacher, that attention transfer always beats logits KD, or that this result will repeat across seeds. Observed aggregate private scores are known, but hidden labels and paired hidden-error counts are not available. The student also exceeds the teacher's recorded clean score, but that comparison is descriptive across saved evaluations, not a controlled proof of mechanism.

**Decision-process correction:** the 97.6% development target was a screening heuristic, not a necessary condition for public improvement. We should have expressed the local evidence as a small, uncertain positive effect with tradeoffs, rather than letting an arbitrary target or worsening probability scores dominate the decision. Conversely, the eventual win does not retroactively justify a calibrated 60–70% confidence claim. Uncertainty was real.

**Testing discipline:** the seven-condition profile and eight additional tests characterized risk; they did not supply independent evidence by multiplication. Predeclare a bounded diagnostic set and stop when further results cannot change the decision. Do not keep testing the same set to seek reassurance. The official test was justified by a fixed differentiated candidate, a near deadline, an expiring slot and explicit user/team allocation—not guaranteed performance.

## Reproduction and verification limits

The maintained workflow and required course-file layout are in the [README](../../README.md). The [verification summary](READABLE_PIPELINE_VERIFICATION.json) records exact same-runtime CPU logit parity for both selected checkpoints on 16 development images, plus sampled augmentation, loss and student-gradient parity. Its source hashes identify the implementation that was checked.

These are bounded refactor checks, not a completed 200-epoch CUDA reproduction or independent model-performance evidence. Original Kaggle sources depend on historical private mounts. Raw checkpoints and data are not distributed in this repository, so a clone alone cannot run exact-checkpoint verification. The teacher notebook is retained under `notebooks/`; its old environment paths require adjustment. Some other team notebooks have unresolved validation/submission mappings and are not pooled into these results.

The original pre-showcase version of this record and retired tooling are preserved locally in ignored `working/`. No new training, hidden-image inspection, submission or private-score-guided tuning was performed for this update.
