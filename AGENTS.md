# Agent instructions

## Purpose and entry points

This is a postcompetition showcase and reproduction repository. Prioritize clear
explanations, maintainable source and faithful verification of recorded results.
Do not treat historical experiment plans as authorization to launch new work.

Read `README.md`, the relevant `src/` modules and `configs/` before making changes.
`src/train.py` is the final attention-student trainer; `src/train_resnet.py` is a
historical SE-ResNet implementation. `kaggle_runs/README.md` describes the retained
execution snapshots. Their contents are historical records, not portable launch
instructions.

## Preservation and scientific accuracy

- Preserve course images, labels, original checkpoints, submission CSVs and raw run
  artifacts. Never quietly edit labels or remove difficult examples.
- Preserve the retained Kaggle snapshots and their recorded source hashes. Make
  maintained code changes in `src/`; document departures from the executed recipe.
- Keep development, public and private scores distinct. Reused development images
  and correlated probes are not independent confirmation. Do not invent missing
  measurements, member contributions or hidden-data explanations.
- The deployed model must be one residual CNN with at most 5,000,000 trainable
  parameters, including its head. Own-data teachers are training-only.
- Training uses only course-supplied images and models originating from random
  initialization or compliant team checkpoints. No external datasets or pretrained
  weights. Hidden images remain prediction-only; no label recovery or hidden-data
  tuning. The selected models use frozen BN, FP32 original/flip mean logits and no
  inference ensemble.
- Keep method/code citations and accurate AI-assistance acknowledgements.

## Validation and operation

Run commands from the repository root in a PyTorch-capable environment:

- `python -m src.train --help`
- `python -m src.predict --help`
- `python -m src.verify`, only when its required course data and original team
  artifacts are available locally.

Use checks appropriate to the change. State missing inputs and verification limits;
do not claim a full CUDA training reproduction from bounded CPU parity checks.
Training and prediction are separate commands. Do not launch training, paid cloud
resources, competition submissions or publication actions without an explicit
request. Use CLI/API; no computer use. Do not spawn subagents unless requested.
Never claim monitoring continues after the agent turn ends.

## Public and local files

Keep secrets, course data, weights and temporary artifacts out of Git. `working/`,
`data/`, `models/` and `output/` are ignored local storage, not off-machine backups.
The public repo must remain understandable without those local directories;
reproduction instructions must identify the additional artifacts required.

Local operational history lives in `working/EXPERIMENT_TRACKER.md` and
`working/PRIVATE_SCORE_PLAN.md`. These are optional historical records, not public
entry points or current launch instructions. The report and its builder are local
under `working/docs/writeup/` and `working/scripts/`.

After relocations, update maintained imports and documentation unless the user
explicitly requests a move without edits. Preserve historical snapshot contents.
