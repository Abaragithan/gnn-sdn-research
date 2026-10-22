# Experiment, Dataset, and Checkpoint Metadata

This note summarizes the dataset layouts, run provenance, evaluation results, and
checkpoint locations established from the repository scripts, saved results, and
historical logs. Historical filesystem paths are distinguished from the paths that
currently exist.

## Dataset roots and split layouts

| Path | Meaning / contents | Status |
|---|---|---|
| `data/all_mixed/` | Original baseline dataset with `train/`, `validation/`, and broad `test/`. Train has seven 10-node topologies; test has 22 topologies spanning 50–300 nodes. | Exists as a real directory; copied from `/home/abaragithan/data/all_mixed/`. |
| `data/all_mixed_v1/` | Teacher/student dataset with normalized `train/`, `validation/`, and `test/` names. Train has 9,972 samples from topologies 10–170 nodes; validation has 2,191 samples; test has 478 samples from 200–300-node topologies. | Exists as a separate real directory. |
| `data/all_mixed_v1/validation_extra/` | The additional 5-topology validation split formerly at `data/all_mixed/validation` before the dataset roots were separated. | Preserved, not used by the recorded teacher/student runs. |
| `data/research/teacher/` | Preserved original-style organization with `train/` (six 10-node topologies), `val/` (one 10-node topology), and `test/` (22 topologies, 50–300 nodes). | Exists inside the repository. Graph/routing payloads match the corresponding 29 topology folders in `data/all_mixed` and `data/all_mixed_v1` (SHA-256 checked). |
| `data/sim_data_v1/` | Separate small generated simulation dataset with `train/`, `validation/`, and `test/`. | Exists; not used by the recorded baseline, teacher, or student results described below. |

### Dataset folder reorganization

The layout was reorganized to keep the original baseline data separate from
the teacher/student experiment data, while retaining each split without
combining or overwriting its contents.

| Before | After | Change |
|---|---|---|
| `data/all_mixed/original/train/` | `data/all_mixed/train/` | Original baseline training split moved to the baseline root. |
| `data/all_mixed/original/validation/` | `data/all_mixed/validation/` | Original baseline validation split moved to the baseline root. |
| `data/all_mixed/original/test/` | `data/all_mixed/test/` | Original broad baseline test split moved to the baseline root. |
| `data/all_mixed/train_teacher/` | `data/all_mixed_v1/train/` | Teacher/student training split moved to the v1 root and renamed to `train`. |
| `data/all_mixed/val_teacher/` | `data/all_mixed_v1/validation/` | Teacher/student validation split moved to the v1 root and renamed to `validation`. |
| `data/all_mixed/test/` (200–300-node OOD split) | `data/all_mixed_v1/test/` | Teacher/student OOD test split moved to the v1 root; its name is unchanged. |
| `data/all_mixed/validation/` (additional 5-topology split) | `data/all_mixed_v1/validation_extra/` | Preserved separately to avoid overwriting or merging it with the teacher validation split. |

Before the reorganization, the original baseline splits were nested under
`data/all_mixed/original/` alongside the teacher/student splits. Afterward,
`data/all_mixed/` contains only the original baseline splits and
`data/all_mixed_v1/` contains the teacher/student splits plus the preserved
extra validation split. The original external source at
`/home/abaragithan/data/all_mixed/` was left untouched; the baseline copy was
verified against it using SHA-256 hashes (907 train, 58 validation, and 396
test files). No split contents were intentionally merged.

### Git-history cross-check

| Commit | Evidence |
|---|---|
| `4d1b3e0` (2026-04-03) | The initial committed teacher scripts read `data/all_mixed/train_teacher`, `val_teacher`, and `test`. |
| `4667525` (2026-10-19) | Adds the student model/evaluation and experiment documentation used as provenance for the student work. |
| `842d9cb` (2026-10-22) | Records the normalized teacher/student script paths (`data/all_mixed_v1/{train,validation,test}`), adds the historical training log and this metadata document, and removes the `.gitignore` rule that excluded `logs/`. |

The committed historical log documents the teacher split construction: it
moves selected 50–170-node topologies and original 10-node training topologies
into `train_teacher`, then moves one topology per size into `val_teacher`.
It records 9,972 train samples, 2,191 validation samples, 478 OOD test samples,
and the teacher evaluation metrics. The current
`data/all_mixed_v1/validation_extra/` is 58 files and is SHA-256-identical,
with matching relative paths, to the current original
`data/all_mixed/validation/`.

Git history does not contain the dataset directories themselves: `data/` is
ignored by Git. Thus commits establish the scripts, logged split-building
commands, results, and later path updates, but cannot independently prove every
physical dataset move. The file identity checks above establish the current
copy relationships, not a complete filesystem audit trail.

### Split paths

Baseline scripts use `data/all_mixed/{train,validation,test}`. Teacher and
student scripts use `data/all_mixed_v1/{train,validation,test}`. The two `test/`
directories intentionally differ: the baseline test spans 50–300 nodes, while
the v1 test is held out to 200–300 nodes.

## Recorded delay experiments

| Experiment | Training split | Validation / checkpoint selection | Evaluation split and recorded result | Checkpoint |
|---|---|---|---|---|
| Baseline Run 1 | Historical `data/all_mixed/train`; 10,000 samples from 10-node-only data. | Historical `data/all_mixed/validation`; training best val MAPE 6.499. A separate 50-sample result (7.474) appears in the log, but its exact invocation/path is not fully clear. | Historical broad `data/all_mixed/test`: 878 raw / 877 usable samples, 50–300 nodes; MAPE 9.1707. | `all_mixed/ckpt_dirs/ckpt_dir_0_10000/03-6.50` |
| Baseline Run 2 | Same historical 10-node training setup. | Historical `data/all_mixed/validation`; training best val MAPE 6.410. Separate 50-sample evaluation MAPE 6.8509 is logged. | Same broad test: 878 samples reported by evaluator; MAPE 8.2249. | `all_mixed/ckpt_dirs/ckpt_dir_0_10000/15-6.41` |
| Teacher | `data/all_mixed_v1/train`; 9,972 samples, topologies 10–170 nodes | `data/all_mixed_v1/validation`; 2,191 samples. Best checkpoint selected at epoch 26; train log val MAPE 4.7139. | `data/all_mixed_v1/test`; 478 samples on 200–300-node topologies; MAPE 5.3734. | `all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26` |
| Student-Scratch | `data/all_mixed_v1/train`; same 9,972-sample split | `data/all_mixed_v1/validation`; evaluated checkpoint val MAPE 10.6455. | `data/all_mixed_v1/test`; same 478-sample 200–300-node test; MAPE 11.9244. | `all_mixed/student/ckpt_dirs_scratch/ckpt-08` |
| Student-KD (alpha=0.5) | `data/all_mixed_v1/train`; teacher checkpoint used to provide distillation predictions | `data/all_mixed_v1/validation`; evaluated checkpoint val MAPE 9.0444. The training result JSON separately reports best val MAPE 9.6439; this discrepancy is unresolved. | `data/all_mixed_v1/test`; same 478-sample 200–300-node test; MAPE 9.8280. | `all_mixed/student/ckpt_dirs_kd_alpha0.5/ckpt-19` |

The baseline test results (9.1707 and 8.2249) use the broad
`data/all_mixed/test` 50–300-node set. They are not directly comparable to
teacher/student results on the narrower `data/all_mixed_v1/test` 200–300-node
set. A fair baseline comparison requires evaluating the baseline checkpoint on
the exact 478-sample v1 OOD test split.

## Script-to-checkpoint provenance

| Purpose | Script(s) associated with recorded result | Checkpoint/output evidence |
|---|---|---|
| Baseline training | `all_mixed/few_shot_all.py` | Baseline logs report 10,000 samples and checkpoints under `ckpt_dirs/ckpt_dir_0_10000/`. |
| Baseline evaluation | `all_mixed/validation_all.py` | Historical logs report `03-6.50` and `15-6.41`; broad test evaluations report 878 samples. |
| Teacher training | `all_mixed/train_teacher.py` | Historical log records saves under `ckpt_dirs_teacher/ckpt_dir_0_9972/`, through `ckpt-26`. Current script reads normalized `data/all_mixed_v1/train` and `validation` paths. |
| Teacher evaluation | `all_mixed/test_teacher.py` | Historical log says `python test_teacher.py`; `all_mixed/results_teacher.json` identifies `ckpt-26`. Current script reads v1 `validation` and `test`. |
| Alternate teacher scripts | `all_mixed/teacher/train.py`, `all_mixed/teacher/evaluate.py` | These use `all_mixed/teacher/ckpt_dirs/...`. No checkpoints were found in that expected location. They are alternate copies; no evidence ties them to the recorded teacher result. Their data paths now point to v1. |
| Student-Scratch training | `all_mixed/student/train_scratch.py` | Checkpoint `ckpt-08` in `student/ckpt_dirs_scratch/`; current script reads v1 `train` and `validation`. |
| Student-KD training | `all_mixed/student/train_kd.py` | Checkpoint `ckpt-19` in `student/ckpt_dirs_kd_alpha0.5/`; KD log confirms teacher checkpoint `ckpt-26`; current script reads v1 `train` and `validation`. |
| Student evaluation | `all_mixed/student/evaluate.py` | `all_mixed/student/results_student.json` records both student checkpoint paths and validation/test metrics; current script reads v1 `validation` and `test`. |

Checkpoint and result evidence is stronger than script file timestamps. A newer
script timestamp alone does not establish whether a script was run.

## Model size metadata

| Model | Verified trainable parameters | Source |
|---|---:|---|
| RouteNet-Fermi teacher / baseline | 27,025 | `all_mixed/evaluation/inference_benchmark_results.json` |
| Student architecture | 4,505 | Same benchmark result; student configured with `state_dim=16`, `iterations=4` |

Older documentation values of approximately 4,945 teacher parameters and 1,241
student parameters are stale relative to the saved benchmark.

## Other experiment

The original RouteNet (not RouteNet-Fermi) log evaluates pretrained checkpoint
`model.ckpt-260380` against GEANT2BW evaluation TFRecords under
`/home/abaragithan/routenet_data/geant2bw/tfrecords/evaluate/`. The available
log establishes the evaluation data path; it does not establish a training
split for that experiment.
