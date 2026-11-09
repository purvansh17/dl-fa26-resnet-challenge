# Selected Kaggle execution snapshots

These sources preserve the historical runs behind the selected submissions.
Use `src/` for the maintained training, prediction and verification commands.

| Directory | Historical role |
|---|---|
| `attention_candidate/` | Original-strength attention student training |
| `attention_beta_pair/` | Half/double-strength comparison, including the selected double-strength student |
| `attention_predictions/` | Original-strength student prediction export |
| `beta_double_predictions/` | Double-strength student prediction export |

`configs/source_provenance.json` pins `attention_beta_pair/worker_double.py` by
SHA-256; `python -m src.verify` uses that source for implementation comparisons.
Snapshot contents were preserved unchanged. Their Kaggle metadata references
private account resources and is not a portable launch configuration.
