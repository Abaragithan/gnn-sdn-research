# Research Plan: Finish by Dec 1

This plan assumes the written report and final presentation are both due Dec 1, 2026, and CPU overnight runs are available. The priority is to complete the missing student OOD evaluation, add limited repeatability evidence, then prepare the report and presentation from verified results.

## Current Evidence

The latest progress summary is dated Oct 23, 2026; check saved outputs before relying on this snapshot.

- Baseline trained on 10-node topologies: historical broad 50–300-node test MAPE 8.2249% (second run; first run 9.1707%); matched 478-sample 200–300-node OOD MAPE 8.4011% using checkpoint `15-6.41`.
- Teacher trained on 10-170-node topologies: validation MAPE 4.71%, OOD test MAPE 5.37% on 200-300-node topologies.
- Student scratch: validation MAPE 10.65%, OOD MAPE 11.9244%. Student KD (alpha=0.5): evaluated validation MAPE 9.0444% (training-reported 9.6439%), OOD MAPE 9.8280%.
- The inference benchmark reports 4,505 student parameters and 34.2 ms mean latency, versus 27,025 parameters and 178.5 ms for the teacher. The progress summary also contains a stale estimate of about 1,241 student parameters; verify and correct it before submission.
- An evaluator exists at `all_mixed/student/evaluate.py`; it evaluates validation and OOD test sets and writes `results_student.json` relative to its working directory.

## Timeline and To-Do

### Oct 20-23: Evaluate Existing Student Checkpoints

- [x] Check for newer results and confirm the intended best scratch and KD checkpoints.
- [ ] Confirm that the 200-300-node test split remains untouched by training or model selection.
- [ ] Record checkpoint paths, seeds, configurations, and expected sample counts.
- [x] Run `all_mixed/student/evaluate.py` on both saved models and save the full results.
- [ ] If checkpoint selection is ambiguous, explicitly evaluate the known best checkpoint before interpreting results.
- [ ] Do not tune models using OOD test results.

### Oct 24-26: Validate and Interpret the Results

- [ ] Compare baseline, teacher, student scratch, and student KD.
- [x] Re-evaluate the baseline checkpoint on the same 478-sample OOD test set as teacher and students; result saved to `all_mixed/results_baseline_ood.json` and log to `all_mixed/baseline_ood.log`.
- [ ] Report MAPE and MAE; include RMSE if useful. Break errors out by topology size as well as overall.
- [ ] Check for missing or invalid predictions and confirm that metric aggregation is comparable across models.
- [ ] Account for the test set's different delay range; do not rely on one aggregate percentage alone.
- [ ] Make an initial results table and report student OOD results honestly, including an unfavorable result if that is what the evaluation shows.

### Oct 27-Nov 9: Add Limited Repeatability

- [ ] Make seeds and checkpoint/output paths configurable if needed.
- [ ] Train at least two additional seeds for student scratch and KD with the same data split and stopping rules.
- [ ] For each run, record seed, selected checkpoint, validation score, OOD score, runtime, and configuration.
- [ ] Keep the OOD test set for final evaluation, not checkpoint selection.
- [ ] If compute limits intervene, keep the existing-checkpoint comparison and disclose the single-seed limitation rather than delaying the report. Additional teacher repeats are optional.

### Nov 10-16: Consolidate Claims and Figures

- [ ] Summarize student repeats with mean and standard deviation where appropriate; preserve individual run values.
- [ ] Create a comparison table for baseline, teacher, student scratch, and student KD covering validation/OOD accuracy and inference cost.
- [ ] Create figures for OOD error by topology size and accuracy versus latency/model size.
- [ ] Verify the student's actual parameter count against the model and benchmark; correct the stale 1,241 claim in `docs/research_progress_summary.md` and any other affected documentation.
- [ ] Ground any architecture novelty claim in the related literature. Otherwise, describe Queue-Link Fusion as the proposed design and report its measured trade-offs.

### Nov 17-24: Write the Research Report

- [ ] Draft the abstract, problem and motivation, related work with complete citations, research questions/contributions, dataset and split, method, experimental protocol, results, discussion, limitations/threats to validity, conclusion, future work, and reproducibility appendix.
- [ ] Clearly distinguish measured results from hypotheses and proposed future work.
- [ ] Include code, dataset provenance, and licensing details as required by the department.
- [ ] Follow the department's required format, page count, and rubric if they differ from this outline.

### Nov 25-29: Prepare and Rehearse the Presentation

- [ ] Build a focused presentation covering the problem, research gap, approach, experimental split, results, efficiency trade-off, limitations, and conclusion.
- [ ] Use the same verified values and figures as the report.
- [ ] Rehearse to the allotted time.
- [ ] Prepare concise answers about OOD split integrity, target-distribution shift and MAPE, repeat count, and the student accuracy/efficiency trade-off.

### Nov 30: Final Audit

- [ ] Confirm the report, slides, and figures open correctly.
- [ ] Cross-check every table and plotted value against saved evaluation outputs.
- [ ] Check citations, captions, units, checkpoint/configuration provenance, and limitations.
- [ ] Remove unsupported claims and retain scripts, configurations, and result files needed to substantiate the reported tables.

### Dec 1: Present

- [ ] Keep conclusions proportional to the evidence.
- [ ] Present the teacher's measured OOD improvement, student compression/latency results, and KD validation improvement.
- [ ] Claim a student OOD benefit only if the held-out evaluation supports it.

## Relevant Files

- `all_mixed/student/evaluate.py` - student validation and OOD evaluation entry point.
- `all_mixed/student/train_scratch.py` and `all_mixed/student/train_kd.py` - student training configuration, seeds, early stopping, and checkpoints.
- `all_mixed/student_delay_model.py` - student architecture; verify parameter count here and against the benchmark.
- `all_mixed/evaluation/inference_benchmark_results.json` - existing latency and model-size evidence.
- `docs/research_progress_summary.md` - experiment history and stale parameter-count claim; update with final verified results.
- `logs/RouteNet-F-Teacher(delay).txt` - split counts, target distributions, and teacher training/evaluation record.
- `docs/routenet_fermi_deep_dive_part1.md` and `docs/routenet_fermi_deep_dive_part2.md` - background material to condense; these do not replace a properly cited related-work section.

## Verification Criteria

1. The evaluator saves `val_teacher` and full `test_ood` results for the intended scratch and KD checkpoints, with checkpoint identity and expected sample counts.
2. Repeat runs use distinct seeds and checkpoint directories; raw per-run values are retained and aggregate results are reported appropriately.
3. Final comparison metrics are checked against saved outputs, and parameter-count and latency claims agree with the benchmark artifact.
4. Report and presentation figures and tables match the verified results, and the presentation is rehearsed to its time limit.

## Scope

- Focus on finishing delay-prediction research and the written report/presentation.
- Unless required by the supervisor, defer jitter/loss extensions, new larger-topology datasets, broad model-family comparisons, and large hyperparameter sweeps until after Dec 1.
- Preserve the OOD set as held-out evaluation; do not use it for model or hyperparameter selection.