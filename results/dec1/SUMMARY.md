
## Problems

T3 benchmark stopped: DatanetException: ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]. See results/dec1/latency_v2_runs_20261122_205401/baseline_default.json.

Resolution (22 Nov 2026): the bundled `DatanetAPI.set_files_to_process` shadows the built-in `tuple`, so its type check rejects valid selections. T3 now copies the selected tuples directly into the API's `_selected_tuple_files` field, preserving the original archive order. Historical API code and checkpoints were not modified. Read-only checks for 200, 240, 260, 280 and 300 nodes all matched T2's first scenario IDs; see `latency_v2_selector_check.log`. No model inference or full benchmark restart was performed by the execution agent. The user subsequently restarted the queue in `latency_v2_runs_20261122_212858/`; it completed all 48 combinations. The failed attempt is preserved.

T3 historical-timing deviation (22 Nov 2026): teacher/default/validation measured **181.6339 ms**, +1.76% versus 178.5 ms (within 20%). Scratch/default/validation measured **41.1798 ms**, +20.41% versus 34.2 ms, slightly outside the 20% bound of 41.04 ms. The cause has not been established. All eight workers and 48 combinations completed successfully; use `latency_v2.json` for the new measurements and report this deviation rather than claiming both reproduction checks passed. No rerun was started to chase the historical value. Full coverage and completion were checked; T4 has not started.

## Problems

T4 preflight stopped: ValueError: Current and frozen scenarios differ. The failed output and log are preserved as `emulation_hop_order_preflight_failed.json` and `emulation_hop_order_preflight_failed.log` in this folder.

Preflight resolution (22 Nov 2026): the only scenario difference was the frozen validation metadata `measurement_s: 60.0`, which is not read by the model input adapter. The diagnostic now verifies equality of every other scenario field, then uses the exact frozen configuration and hash. Checkpoint hashes and measurement-source provenance are also checked. The diagnostic was rerun; its outputs are ready for R3 review in `emulation_hop_order.json` and `.log`. T4 ranking/choice conclusions have not been added here pending R3.

## Emulation under the training hop order — R3 approved, 23 Nov 2026

With the training-time (reference) hop order, all three models still rank the candidates **1, 0, 2**, still select **candidate 1**, and pass all physical checks. Foreground predictions change by at most **0.21 ms** between orders. True-hop-order predictions reproduce the frozen predictions exactly (maximum difference **0.0 ms**). The emulation's ranking and selected candidate therefore survive the change in input-order convention for this fixed six-switch scenario.

| Model | Mean foreground trial APE, true hop order (%) | Mean foreground trial APE, training/ref order (%) |
|---|---:|---:|
| teacher | 7.198059 | 7.246069 |
| scratch | 5.280694 | 5.283797 |
| kd | 6.770033 | 6.739159 |

APEs use the 15 measured foreground trial means (five per candidate), not fallback values, training seeds or all background flows. Results and source/checkpoint hashes: `results/dec1/emulation_hop_order.json`; log: `results/dec1/emulation_hop_order.log`. The JSON's `review_status: pending_R3` records its generation-time status; approval is documented in the execution plan's R3 review result dated 23 Nov.

Report impact: the statement in §3.7 that the effect on the emulation's predictions was not measured, and the matching threat/future-work items, are resolved. Together with T2, the effect was measured and small offline and in this emulation; this does not establish general routing-decision accuracy.

## T5 student-training smoke checks — passed, R4 pending (23 Nov 2026)

The reviewer-provided `all_mixed/student/train_student_v2.py` ran unchanged. Three seed-99 runs each completed 50 training steps, one epoch and five validation scenarios. Scratch created the shared init; both KD runs loaded it. All initial-file SHA-256 hashes matched and every first-step hard loss was **31.252909**. Seeds 1 and 2 were not used.

| Run | Hard loss | Soft loss | Validation MAPE (%) |
|---|---:|---:|---:|
| smoke_scratch | 23.478565216 | 0.000000000 | 18.998365520 |
| smoke_kdrel | 23.478540421 | 23.514001846 | 18.998277313 |
| smoke_kdmse | 23.478630066 | 0.316480219 | 18.998422465 |

Independent T2 evaluator on scratch `ckpt-01` and the same five validation scenarios: **18.998365520112%**, matching training validation MAPE **18.998365520112%** exactly (difference **0.0**, tolerance 0.01 points). Relative KD soft loss is on the hard-loss scale; original MSE is much smaller; scratch soft loss is zero. All three T5 smoke gates passed. These short runs are functional checks, not final accuracy evidence.

R4 review evidence is preserved in `results/dec1/t5_smoke/`: all three configs, histories, best records and complete logs; validation JSONL/metadata/log/summary; and `checks.json` with gate numbers and SHA-256 manifest. Per the plan, the original smoke folders, seed-99 init and original validation smoke files are removed after this preservation; checkpoint paths in copied metadata remain historical provenance. Full T6 training has not started. Await R4 review.

Warning/logging note for R4: each smoke run emitted 12 sparse-IndexedSlices-to-dense UserWarnings. All three logs together occupy about 42 KiB; warning text accounts for approximately 23 KiB. The original logs remain unfiltered and their hashes are unchanged. The user requested less repetitive logging. An optional future command prefix, `PYTHONWARNINGS='ignore:Converting sparse IndexedSlices:UserWarning:tensorflow.python.framework.indexed_slices'`, selectively hides this exact warning without training-code edits; a synthetic filter check preserved an unrelated warning. No training run has used this filter. It hides output only and does not reduce dense-gradient memory consumption.

## R4 approval and logging deviation — 23 Nov 2026

R4 approved all three T5 gates. Reviewer commit `94eacee` made only logging changes to `train_student_v2.py`: imported `warnings`, filtered the repeated `Converting sparse IndexedSlices` message inside `main`, and passed `verbose=2` to `fit` for epoch-level logging. The diff was checked; training, losses, validation, initialization and checkpoint saving are unchanged. The reviewer explicitly waived another smoke run. No external `PYTHONWARNINGS` prefix is needed for T6.

The current source hash differs from the pre-review hash in `t5_smoke/checks.json` because of these logging changes. Keep the archived smoke evidence and its original manifest unchanged; real-run configs record the new git commit. R4 is complete. The user subsequently ran T6's first pair; see the 24 Nov hand-off below. The execution agent has not launched full training.

## Controlled KD study — seed-1 interim review (24 Nov 2026)

Two required T6 runs completed normally with finite recorded losses/metrics. These are **validation-only** results; no new test-set evaluation was performed. Select the checkpoints listed in `best.json`, not each run's last weights unless they coincide.

| Run | Best validation MAPE (%) | Best epoch / epochs run | Selected checkpoint | Hours | Stop reason |
|---|---:|---:|---|---:|---|
| scratch_s1 | 10.153210530 | 21 / 26 | `runs/scratch_s1/ckpt-21` | 10.17 | Early stopping |
| kdrel_s1 | 9.851046374 | 30 / 30 | `runs/kdrel_s1/ckpt-30` | 12.27 | Epoch limit |

KD − scratch is **−0.302164157 validation MAPE points**, or **2.976% relative error reduction**. KD improved validation performance for this seed. This does not establish a repeatable or statistically significant effect, and the test outcome remains unknown. The second paired seed is still required. KD's best score occurred at the final allowed epoch; the existing 30-epoch cap and patience remain unchanged.

Both runs use seed/data-seed 1, the same saved initial student weights, identical first-step hard loss (**33.599453**), the same code (`94eacee`), 9,972 steps per epoch and flow-weighted validation selection. Scratch recorded `init_status: created`; KD recorded `loaded`. The KD teacher checkpoint hashes match T2's fixed `ckpt-26`. Selected checkpoint files are present locally. Exact values, settings and hashes are in `results/dec1/t6_seed1_review.json`; configs, histories, best records and logs are under the two run folders. Checkpoint binaries must remain uncommitted.

Scheduling deviation: scratch took **10.17 h** rather than approximately 5.5 h; KD took **12.27 h** rather than approximately 10 h. The pair took **22.44 h** of training/validation, with scratch ending 23 Nov at 16:47 and KD ending 24 Nov at 05:35 (Asia/Colombo). Allow roughly 22–24 h for the remaining required pair and 12–13 h for the optional MSE run if similar conditions hold. These are estimates; the reviewer should reassess the queue against the fixed T8 start/number-lock deadlines. Prioritize the required seed-2 pair over the MSE replication.

Current status: the user requested this interim review before continuing. `scratch_s2`, `kdrel_s2` and `kdmse_s1` remain pending in the plan; no additional run has been started by the execution agent. No training settings, code or checkpoints were changed during this results check.

## Controlled KD study — two-pair review (25 Nov 2026)

This supersedes the seed-1-only status above. All four required T6 jobs completed normally. **The following are validation results, not final test results.** No new test evaluation has been performed.

| Run | Best flow-weighted validation MAPE (%) | Best epoch / epochs run | Selected checkpoint | Hours |
|---|---:|---:|---|---:|
| scratch_s1 | 10.153210530 | 21 / 26 | `runs/scratch_s1/ckpt-21` | 10.17 |
| kdrel_s1 | 9.851046374 | 30 / 30 | `runs/kdrel_s1/ckpt-30` | 12.27 |
| scratch_s2 | 10.014834048 | 15 / 20 | `runs/scratch_s2/ckpt-15` | 7.77 |
| kdrel_s2 | 11.918671162 | 4 / 9 | `runs/kdrel_s2/ckpt-04` | 3.65 |

Paired KD − scratch validation differences are **−0.302164157 points for seed 1** and **+1.903837114 points for seed 2**, with mean **+0.800836479 points**. Mean best-validation MAPE is **10.084022289% for scratch** and **10.884858768% for KD**. KD improved validation in **1 of 2** pairs and was worse on average. There is no consistent validation benefit under this protocol. Do not claim statistical significance or infer the final test outcome from these two seeds.

All recorded history values are finite. Selected checkpoint files exist and `best.json` matches each run's validation minimum. Each scratch/KD pair has identical initial hashes and first-step hard losses (seed 1: **33.599453**; seed 2: **34.021622**), matched data order seed and stopping settings. Training-source hashes are identical across the source commits used by the runs. Both KD jobs used the fixed teacher verified in T2. Seed 2's unfavorable KD result is valid experimental evidence; no technical failure was detected in these checks.

Scratch seed 2 stopped after epoch 20 with its best at 15; KD seed 2 stopped after epoch 9 with its best at 4, following five non-improving validation epochs. No epochs, patience, loss weights or seeds were changed to improve the outcome. Preserve and evaluate all four selected models at T8.

The four runs total **33.85 h** of training/validation, excluding gaps. Seed-2 runs finished on 24 Nov at **14:06** and 25 Nov at **00:37**, Asia/Colombo. The planned `kdmse_s1` remains pending and could take roughly 12–13 h if it reaches 30 epochs. Review its priority against the remaining schedule; keep T8's latest start of 27 Nov, 20:00 and the number lock of 28 Nov, 08:00.

Exact review data and provenance: `results/dec1/t6_paired_validation_review.json`. New seed-2 configs, histories, best records and logs are in their run folders; checkpoint binaries must not be committed. The user requested review before proceeding. No further training, T7 experiment or T8 evaluation was started by the execution agent. The report agent should retain the mixed validation outcome; final claims about test accuracy await T8.

## T6 three-pair hand-off — request T8 before MSE (26 Nov 2026)

**Current status: all six scratch/MAPE-KD runs finished; review of the requested order change is pending.** This supersedes earlier completion/status snapshots. The user ran seed 3 one job at a time, rather than using the proposed chained queue. No MSE training or T8 evaluation was launched during this check.

| Run | Best flow-weighted validation MAPE (%) | Best epoch / epochs completed | Selected checkpoint | Hours |
|---|---:|---:|---|---:|
| scratch_s1 | 10.153210530 | 21 / 26 |`runs/scratch_s1/ckpt-21` | 10.17 |
| kdrel_s1 | 9.851046374 | 30 / 30 |`runs/kdrel_s1/ckpt-30` | 12.27 |
| scratch_s2 | 10.014834048 | 15 / 20 |`runs/scratch_s2/ckpt-15` | 7.77 |
| kdrel_s2 | 11.918671162 | 4 / 9 |`runs/kdrel_s2/ckpt-04` | 3.65 |
| scratch_s3 | 10.162595209 | 28 / 30 |`runs/scratch_s3/ckpt-28` | 11.66 |
| kdrel_s3 | 9.958502915 | 30 / 30 |`runs/kdrel_s3/ckpt-30` | 12.22 |

Paired KD − scratch validation differences: seed 1 **−0.302164157**, seed 2 **+1.903837114**, seed 3 **-0.204092294** percentage points. KD is better in **2 of 3** seeds, but mean best-validation MAPE is **10.110213262% scratch** versus **10.576073483% KD**, with mean paired difference **+0.465860221 points** (KD worse on average). Results remain mixed; no consistent KD benefit, statistical significance or final test improvement is established.

**Seed-3 checks PASS:** both runs completed 30 epochs normally; every recorded history metric is finite; `best.json` matches each validation minimum; selected index/data files exist and were hashed. Initial-weight hashes, first-step hard losses (**32.87072**), data seeds and stopping settings match within the pair. Training source matches the earlier pairs, and KD uses the same fixed teacher hashes. Scratch's best is epoch 28; KD's best is epoch 30. Do not extend the training cap. Seed-3 scratch/KD took **11.66 / 12.22 h**, respectively. Exact values, timestamps, hashes and checks: `results/dec1/t6_three_pair_validation_review.json`. Original two-pair evidence is preserved unchanged.

**Request to the reviewing agent:** may we start **T8 now for all six completed students**, then run the already-planned **`kdmse_s1` afterward**, using its fixed settings (`--mode kd --soft-loss mse --alpha 0.5 --seed 1`, 30 epochs, patience 5, existing seed-1 init and teacher)? After MSE finishes, evaluate its validation-selected `best.json` checkpoint with the same T8 test/ref and validation/ref protocol and add its results; existing completed evaluations need not be repeated. Please confirm this order and whether MSE can fit the final deadline or must be labeled follow-up.

If approved, freeze MSE settings before T8 and run it regardless of whether the first T8 results favor KD. Do not tune settings or decide whether to include MSE using test outcomes. Report all six completed runs, including unfavorable seed 2. **This request is not yet reviewer approval.** MSE remains planned, not silently removed. T7 ablation stays dropped. The latest approved T8 cutoff remains **27 Nov, 12:00 Asia/Colombo**, with number lock **28 Nov, 08:00**, unless explicitly revised by review. If MSE misses the agreed deadline, mark it as a deferred follow-up and report the limitation.

**Evidence for user commit/push:** this plan, `results/dec1/SUMMARY.md`, `results/dec1/t6_three_pair_validation_review.json`, and the seed-3 run folders' `config.json`, `history.json`, `best.json`, `train.log`. Keep checkpoint binaries and shared init files local and uncommitted. No commit/push was made by the execution agent.

## T8 six-run evaluation completed — 26 Nov 2026; MSE pending

Test and validation evaluation both finished normally: **478 unique test scenarios and 2,191 unique validation scenarios**, all six selected checkpoints under reference order. Every model covers 31,028,940 test flows. Metadata checkpoint paths and hashes match the training review; all predictions are finite and nonnegative. **All six validation reproduction gates PASS** (absolute differences below 0.01 MAPE points; maximum 3.55e-15 points). Use validation-selected `best.json` checkpoints throughout; no test-based model selection was performed.

| Run | Validation MAPE (%) | Test MAPE (%) | Test MAE (ms) | Best epoch / epochs run | Training + validation hours |
|---|---:|---:|---:|---:|---:|
| scratch_s1 | 10.153210530 | 11.434581472 | 6.649702 | 21 / 26 | 10.17 |
| kdrel_s1 | 9.851046374 | 10.879031757 | 6.399950 | 30 / 30 | 12.27 |
| scratch_s2 | 10.014834048 | 11.162406582 | 6.972258 | 15 / 20 | 7.77 |
| kdrel_s2 | 11.918671162 | 13.263121811 | 7.373957 | 4 / 9 | 3.65 |
| scratch_s3 | 10.162595209 | 11.091398643 | 6.654270 | 28 / 30 | 11.66 |
| kdrel_s3 | 9.958502915 | 11.047944969 | 6.679723 | 30 / 30 | 12.22 |

| Paired seed | Test KD − scratch MAPE (points) | Direction |
|---|---:|---|
| 1 | -0.555549716 | KD better |
| 2 | +2.100715229 | KD worse |
| 3 | -0.043453674 | KD better |
| Mean | **+0.500570613** | **KD worse on average** |

Mean test MAPE: **11.229462232% scratch**, **11.730032845% KD**. KD is better on test in **2 of 3 seeds**, by 0.556 and 0.043 points, but worse in seed 2 by 2.101 points. **No consistent KD benefit is shown.** No significance claims; the three paired outcomes are all retained. Both full-length KD runs selected epoch 30, the agreed cap; this does not justify selective extensions or explain causality. The validation pattern is similar (two small gains, one larger loss), but test results are reported separately.

Per-size test MAPE (%):

| Run | 200 nodes | 240 nodes | 260 nodes | 280 nodes | 300 nodes |
|---|---:|---:|---:|---:|---:|
| scratch_s1 | 11.250612 | 10.639297 | 11.693034 | 11.961790 | 11.888313 |
| kdrel_s1 | 10.831633 | 9.717665 | 11.275720 | 11.603842 | 11.467939 |
| scratch_s2 | 11.135934 | 9.946120 | 11.623325 | 11.868735 | 11.780547 |
| kdrel_s2 | 13.345527 | 11.646176 | 13.847111 | 14.203813 | 14.052573 |
| scratch_s3 | 11.093810 | 9.806911 | 11.448420 | 11.913542 | 11.761367 |
| kdrel_s3 | 11.004372 | 9.885992 | 11.470855 | 11.769735 | 11.619063 |

Evidence: `eval_v2_test.jsonl`, `eval_v2_test.meta.json`, `eval_v2_test_summary.json`, corresponding `eval_v2_val` files, shared `eval_v2.log`, and `eval_v2_checks.json` (coverage, checkpoint provenance, validation gates, paired test differences and artifact hashes). The existing T2/T3/T4 evidence remains the source for historical accuracy, timing and emulation claims. Do not use T8 timing fields as a new latency benchmark.

**Scheduling deviation / next step:** the reviewer proposed parallel MSE training, but the user explicitly chose to finish T8 first. MSE has not started; no concurrent training was launched by the execution agent. The planned MSE settings remain frozen (seed 1, MSE soft loss, alpha 0.5, same saved initialization/teacher, epoch cap 30 and patience 5), independent of these test outcomes. Start it manually afterward; if finished by **27 Nov, 20:00 Asia/Colombo**, evaluate its best checkpoint separately on test and validation and append all results regardless of direction. Otherwise label it deferred follow-up. Push the six-run hand-off without waiting for MSE; number lock remains **28 Nov, 08:00**.

**Report impact:** replace any claim of a repeatable or controlled KD improvement with the three paired test differences and the higher KD mean. Preserve the separate historical single-run result; it does not override this controlled study. Compression/forward-time evidence from T3 remains separate from KD's accuracy effect. MSE has not yet been evaluated and cannot be used to support a loss-function conclusion. The report files were not modified by the execution agent.

## Final seven-run hand-off — MSE completed, 26 Nov 2026

**All planned training and evaluations are complete; final reviewer/report integration and user commit/push remain.** This supersedes the earlier MSE-pending snapshots. T7 ablation remains dropped. No further training or evaluation is planned.

MSE seed 1 completed **15 epochs in 6.19 h**, stopping after five non-improving epochs. Its validation-selected best is **epoch 10**, `runs/kdmse_s1/ckpt-10`. Same seed-1 initial hashes and first-step hard loss (**33.599453**), same fixed teacher, unchanged stopping/data settings and training source all verified. Best checkpoint files exist and their hashes match both evaluation metadata files. All recorded training metrics are finite.

MSE evaluation covers **478 test scenarios / 31,028,940 test flows** and **2,191 validation scenarios**, in exactly the original T8 scenario order. Predictions are finite and nonnegative. Validation pooled MAPE **10.281098880092%** matches `best.json` **10.281098880092%** (difference 1.78e-15 points; tolerance 0.01): **PASS**.

Final controlled-study table (all selected by validation, reference input order):

| Run | Validation MAPE (%) | Test MAPE (%) | Test MAE (ms) | Best epoch / epochs run | Training + validation hours |
|---|---:|---:|---:|---:|---:|
| scratch_s1 | 10.153210530 | 11.434581472 | 6.649702 | 21 / 26 | 10.17 |
| kdrel_s1 | 9.851046374 | 10.879031757 | 6.399950 | 30 / 30 | 12.27 |
| scratch_s2 | 10.014834048 | 11.162406582 | 6.972258 | 15 / 20 | 7.77 |
| kdrel_s2 | 11.918671162 | 13.263121811 | 7.373957 | 4 / 9 | 3.65 |
| scratch_s3 | 10.162595209 | 11.091398643 | 6.654270 | 28 / 30 | 11.66 |
| kdrel_s3 | 9.958502915 | 11.047944969 | 6.679723 | 30 / 30 | 12.22 |
| kdmse_s1 | 10.281098880 | 11.355698164 | 6.897061 | 10 / 15 | 6.19 |

MSE seed-1 test difference from scratch: **-0.078883308 points** (a small improvement); from MAPE KD: **+0.476666408 points** (worse). Validation differences are **+0.127888350** versus scratch and **+0.430052507** versus MAPE KD (both worse). This one MSE run does not establish repeatability or a better distillation loss. Its best-epoch hard loss was **7.616189480**, soft MSE **0.176013827**: alpha 0.5 does not make the two terms equal in magnitude, and switching loss changes the effective balance. No tuning or extra runs were performed based on test results.

MSE test MAPE by topology size:

| 200 nodes | 240 nodes | 260 nodes | 280 nodes | 300 nodes |
|---:|---:|---:|---:|---:|
| 11.321431% | 10.075367% | 11.794976% | 12.137347% | 12.010471% |

**Final interpretation:** retain the three-pair MAPE KD finding: test KD − scratch **−0.555549716 / +2.100715229 / −0.043453674 points**, mean **+0.500570613 points**. KD improved two seeds but was worse on average; **no consistent KD benefit is shown**. The single MSE run does not change this conclusion. No significance claim. Compression/forward-speed findings remain supported separately by T3, not attributed to KD.

**Problems/deviations and report impact:** execution was sequential (six-run T8, then MSE training/evaluation) at the user's request, rather than parallel as the reviewer proposed. Training/evaluation finished normally; no MSE restart or protocol change. MSE is included despite its unfavorable comparisons. Historical single-run findings remain separate from the controlled study. Update the report with all seven rows, three paired MAPE-KD differences, the single-seed MSE result and its scale/replication limitation. The execution agent did not modify report files.

**Final review request:** verify the seven-run evidence, interpretation, and closure of T6/T8 execution; integrate final results into the report. Evidence: `eval_v2_kdmse_checks.json`, MSE test/validation JSONL, metadata and summary files, `eval_v2_kdmse.log`, and `runs/kdmse_s1/{config.json,history.json,best.json,train.log}`. Existing `eval_v2_checks.json` and six-run artifacts remain unchanged. Commit/push JSON/log/documentation evidence only; checkpoint binaries/shared init stay local and ignored. No commit or push was made by the execution agent.
