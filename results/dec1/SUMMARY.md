
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
