# Mininet routing experiment

The completed study is a controlled six-switch FIFO/CBR experiment on one laptop.
Five trials per candidate path are complete. Commands below are for reproduction;
there is no need to repeat completed trials now.

## 1. Read the five Python files in this order

| File | Responsibility | Start here |
|---|---|---|
| `settings.py` | Topology, traffic, units and checkpoints | `scenario()` |
| `run.py` | Create network, measure, save and clean up | `main()` |
| `model_worker.py` | Build tensors, check predictions, load one model | `build_inputs()`, `physical_checks()`, `serve()` |
| `controller.py` | Choose a candidate and install OpenFlow rules | `API.decision()`, `install()` |
| `traffic.py` | Send UDP and record timestamps | `send()`, `receive()` |

```text
settings -> model predicts three candidates -> controller installs selected route
         -> traffic sender/receiver -> packet measurements and switch counters
```

`run.py` follows `start() -> shape() -> measure() -> close()`. Most runner code
records measurements, verifies forwarding and handles cleanup. These checks
are part of the experiment. TensorFlow uses its existing environment; Ryu uses
an isolated `.venv`. `requirements.txt` records compatible Ryu dependencies.

## 2. Fixed configuration

- Six switches, one host each, 30 ordered host-pair flows.
- Foreground: host 0 to host 5. Background routes stay fixed.
- Candidates: **0–1–5**, **0–4–5**, **0–2–3–5**.
- Inter-switch capacity: 20,000 bits/s; access links unshaped.
- FIFO: 32 packets = 32,000 bits. No configured propagation delay.
- Frame: 125 bytes = 14 Ethernet + 20 IPv4 + 8 UDP + 83 payload bytes;
  excludes Ethernet FCS, preamble and inter-frame gap.
- Constant-rate traffic, scaled to 60% busiest-link utilization under reference
  shortest-path routing. Packet phase offsets repeat across trials.
- Two seconds warm-up, followed by measurement and queue drainage.
- Historical checkpoints: teacher `ckpt-26`, scratch `ckpt-08`, KD `ckpt-19`.
  Exact paths are in `settings.py`; checkpoint hashes are saved in results.

The default duration remains the original 10-second pilot. Specify `--seconds 60`
for the completed longer protocol. No training is performed here.

## 3. Reproduction commands

Run from the repository root, one command at a time. Each `--name` must be new;
existing outputs are preserved. Enter sudo passwords only in your terminal.

### Model-driven routing

```bash
sudo /usr/sbin/python mininet_experiment/run.py --method teacher --seconds 60 --name teacher_60_01
sudo /usr/sbin/python mininet_experiment/run.py --method scratch --seconds 60 --name scratch_60_01
sudo /usr/sbin/python mininet_experiment/run.py --method kd --seconds 60 --name kd_60_01
```

Model runs check predictions before starting Mininet. Invalid predictions save
failure details and stop the run. Preflight/startup time is separate from decision
latency. The worker is checked again before installing a route.

### Measure fixed candidates

```bash
sudo /usr/sbin/python mininet_experiment/run.py --method shortest --candidate 0 --seconds 60 --name path0_new
sudo /usr/sbin/python mininet_experiment/run.py --method shortest --candidate 1 --seconds 60 --name path1_new
sudo /usr/sbin/python mininet_experiment/run.py --method shortest --candidate 2 --seconds 60 --name path2_new
```

`--candidate` forces a route for measurement and is allowed only with `shortest`.
Without it, `shortest` chooses candidate 0. Forced candidates 1/2 are not shortest-
path baseline decisions. Results explicitly record `forced_candidate`.

### Check a model without Mininet or sudo

```bash
/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python mininet_experiment/model_worker.py teacher --check
```

Optional `--scenario path.json` loads a saved scenario. The original Abilene JSON
is retained for diagnosis; all three model methods fail its physical checks.

## 4. Measurements and correctness

- Processing: input preparation, inference for three candidates, communication,
  installation through barriers, total decision-to-install and startup/preflight.
- Network: mean/p95 one-way delay, packet loss and received payload goodput.
- Verification: offered rates, pacing, queues, frame sizes, route validity,
  installed rules, path counters, duplicate packets and OpenFlow errors.
- Prediction bounds: non-finite values, negative queued bits, overfull FIFO,
  total delay below serialization or above serialization plus full-queue waiting.
  Tolerances: 0.001 bit and 1e-8 second. Predictions are never clipped.

Physical bounds apply to this single-FIFO-per-link experiment. Passing does not
prove accuracy. Delay includes Linux/userspace scheduling; hosts share a monotonic
clock. Measured outcomes never enter prediction inputs. Delay/loss follow packets
scheduled within the measurement window, including arrivals during drainage.
Goodput counts payload arriving within the window.

The adapter preserves route hop order and verifies reverse tensor connectivity.
That differs from historical converter ordering; original training/evaluation
files remain unchanged. The reference comparison is retained with diagnostics.

## 5. Where evidence is saved

Each `outputs/<name>/` contains scenario/result JSON, service logs, packet CSVs,
flow dumps and queue counters. Model runs also record model identity, predictions,
timings and physical checks. Failed runs retain their evidence.

| Record | Contents |
|---|---|
| `outputs/six_validation_predictions.json` | Frozen predictions, hashes and 30-pass inference timings |
| `outputs/six_validation_summary.json` | All 15 trial sources, statistics and result hashes |
| `outputs/author_real_traffic/` | Original author code/checkpoint, comparison and predictions |
| `outputs/source_archive_before_cleanup/` | Original diagnostic scripts and complete investigation notes |

The source archive preserves `audit.py`, `prepare_abilene.py` and the previous
long README. They are historical reproduction tools, not dependencies of the
five runtime files. To run an archived tool, use the TensorFlow Python and set
`PYTHONPATH=mininet_experiment` from the repository root. Use new output names.
`outputs/` is ignored by Git; preserve it separately with research backups,
including the diagnostic source archive and downloaded author files.

## 6. Completed six-switch results

All 15 fixed-path trials passed with the same scenario hash. Every trial received
all 180 foreground packets: 900 per candidate, 2,700 total, 0% observed loss.
Repetition orders were 0/1/2, 1/2/0, 2/0/1, 0/2/1 and 1/0/2. First-trial names
are `six_validation_path_<candidate>`; later names are
`six_validation_rep<repetition>_path_<candidate>`.

| Candidate | Mean delay, ms | SD of five trial means, ms | Mean of trial p95 values, ms |
|---|---:|---:|---:|
| 0 | 123.785 | 0.016 | 169.144 |
| 1 | 112.929 | 0.019 | 148.144 |
| 2 | 196.331 | 0.099 | 296.375 |

Candidate 1 was best in all five repetitions, with 8.770% lower mean delay than
candidate 0 (paired-reduction SD: 0.023 percentage points). Both have two hops:
this compares one deterministic shortest-path tie choice with delay selection,
not every possible shortest-path policy. The p95 column averages trial percentiles.

| Model | Mean APE across 15 foreground trial means | Inference mean for three candidates, ms |
|---|---:|---:|
| Teacher | 7.20% | 7.55 |
| Scratch | 5.28% | 4.08 |
| KD | 6.77% | 3.62 |

All models selected candidate 1. Inference used 30 passes after five warm-ups,
with prepared inputs; preprocessing and installation are excluded. These network
trials did not measure repeated total model decision-to-install time.

This is one deterministic traffic configuration. Results support a repeatable
controlled pilot, not arbitrary real-traffic accuracy, a KD advantage or a
production real-time guarantee. Network repetitions are not training seeds.

## 7. Abilene investigation retained

Our all-mixed teacher failed at Abilene's larger traffic scale. Tracing found
large embeddings, saturated GRU gates and impossible queue predictions. Restoring
historical route order did not fix it. Uniform scaling cannot fit every demand
and capacity into observed training ranges while retaining their ratios.

On ten original Abilene samples (1,320 flows), the authors' unchanged pipeline
gave 3.81% MAPE with their real-traffic checkpoint and 12,636.82% with ours.
Their checkpoint still predicted 38.56 ms on our changed CBR input for a route
measured at 0.804 ms, and failed some physical checks. Their trace encoding differs
from actual CBR features. Diagnostic feature changes were not adopted as a fix.

This demonstrates checkpoint/input-domain incompatibility. The
[authors' paper](https://arxiv.org/html/2212.12070#S5.SS3.SSS2) adapted a checkpoint
on 200 GEANT samples. Our comparison is a small diagnostic, not full reproduction.
Pinned source hashes, predictions and details are preserved in the result folders
and the archived full README. No research model weights were changed.

## 8. Environment and cleanup

Ryu uses Python 3.9 in `.venv`; Mininet uses `/usr/sbin/python`; TensorFlow uses
`routenet-fermi` as configured in `settings.py`. Loopback ports 6653/18080/18878
must be free. Port 6653 is compatible with this host's Fedora SELinux policy.

The runner uses unique switches, explicit rules/static ARP, refuses conflicting
bridge/controller connections, and stops its own resources. Do not use global
`mn -c`: unrelated OVS bridges may exist. Wait for `Saved: ...`, then inspect
`accepted` and `failures` in `result.json`.

## 9. Handoff for the agent updating the report

> **Status (22 Nov 2026): done.** The report now contains this material. Remaining
> emulation work for Dec 1 (re-scoring with the training hop order) is task T4
> in [`docs/dec1_execution_plan.md`](../docs/dec1_execution_plan.md).

**Scope:** incorporate the completed findings and limitations below into the
current LaTeX report, whose entry point is `report/main.tex`. This README update
does not itself change the report. Older `report_source.md`, `evidence_manifest.json`
and `report/README.md` are absent from this checkout; use the actual LaTeX files.

Read Sections 2, 4, 6 and 7 before editing. Preserve the historical offline
all-mixed results and their original checkpoint/protocol descriptions. The new
deployment metrics must be labelled as a separate experiment, not replacements
for offline OOD metrics. Do not introduce new experiments or invented results.

### Where the report needs changes

| Existing file | Required update |
|---|---|
| `report/frontmatter/abstract.tex` | Briefly add controlled SDN integration, repeated candidate-route validation and lower student inference cost; retain limited scope. |
| `report/chapters/ch1_introduction.tex` | Distinguish the original deployment objective from the completed controlled emulation. |
| `report/chapters/ch3_methodology.tex` | Describe separate Mininet/Ryu/model processes, candidate scoring, fixed background routes, explicit rules and physical checks. |
| `report/chapters/ch4_experimental_setup.tex` | Add the configuration in Section 2, fixed checkpoints, 60-second measurement, warm-up/drainage, five repetitions, rotated order and evidence sources. Qualify existing statements that all experiments use all-mixed. |
| `report/chapters/ch5_results.tex` | Add the Section 6 tables, distinguish network delay from computation, discuss the Abilene diagnostic and expand limitations. |
| `report/chapters/ch6_conclusion.tex` | Replace blanket statements that decision-level/controller evaluation is entirely future work with the completed pilot and its remaining limitations. |
| `report/chapters/appendices.tex` | Add reproduction commands and artifact locations if appropriate; keep the chapter structure and departmental format. |
| `report/references.bib` | Check existing RouteNet-Fermi entries before adding citations; cite the paper's real-traffic adaptation and official code. |

### Required interpretation of the completed result

- Fifteen accepted **fixed-path** trials: three candidates, five repetitions,
  one topology/load/deterministic traffic schedule. These are network repetitions.
- Frozen model predictions rank paths 1, 0, 2, matching measured mean-delay
  ranking in each repetition. The 5/5 agreement repeats the same model choice
  against repeated measurements; it is not five independent traffic scenarios.
- Candidate 1 mean delay is 112.929 ms versus candidate 0's 123.785 ms: average
  paired reduction 8.770%, SD 0.023 percentage points. Both paths have two hops.
  Call candidate 0 the deterministic hop-count baseline, not all shortest paths.
- All 2,700 measured foreground packets arrived. Report **zero observed loss**,
  not a guarantee. SD in the table is across trial means; the p95 column is the
  average of trial p95 values, not a pooled packet percentile.
- The 7.20%/5.28%/6.77% teacher/scratch/KD errors average percentage errors
  against the **15 foreground trial means**. They are not full-dataset MAPEs.
  Scratch being closer here does not establish general superiority over teacher/KD.
- The 7.55/4.08/3.62 ms inference values are means of 30 passes, three candidates
  per pass, after five warm-ups. Student speedups are about 1.85x/2.09x for this
  input. Exclude preprocessing, communication, preflight and installation.
  Do not mix these with the earlier 5.2x single-sample offline benchmark.
- Earlier model-driven pilots demonstrated Ryu installation of the chosen route.
  The 15 subsequent trials forced each path to establish its measured quality.
  They did not execute 15 new model-driven decisions or repeatedly measure total
  model decision-to-install latency. Scratch/KD timing differences do not prove
  a KD-specific speed advantage: their architecture is identical.

### Required limitations and negative findings

Report this as a **controlled SDN emulation proof of concept**, not production
deployment validation. It has six switches, fixed-size CBR UDP, FIFO queues,
fixed background traffic, no configured propagation delay and shared laptop
resources. Repeating a deterministic schedule measures repeatability, not
robustness across traffic conditions. The measured best route is best among
three candidates for this foreground flow; it is not a globally optimal routing.

Abilene remains unresolved. Separate its initial accepted operational runs from
the later prediction-validity findings: operational acceptance did not establish
accurate/physical predictions. The new guard blocks negative/overfull queue
predictions and impossible totals, rather than clipping or repairing them.

Keep the author-checkpoint comparison explicit: the unchanged original pipeline
on ten preselected Abilene samples (1,320 flows) yielded 3.81% MAPE with authors'
`real_traffic/ckpt_dir/01-5.20`, versus 12,636.82% with our all-mixed `ckpt-26`.
Model definitions/normalization matched; learned weights differed. The authors'
weights also failed to validate our adapted CBR scenario (38.56 ms predicted for
the selected route, 0.804 ms measured). Diagnostic feature interventions were
not applied as fixes. The [paper](https://arxiv.org/html/2212.12070#S5.SS3.SSS2)
fine-tuned on 200 GEANT samples for its real-traffic experiment; do not imply
that it used our unchanged all-mixed checkpoint. Our ten-sample comparison is
not a reproduction of the paper's full reported test score.

The supported explanation is checkpoint/input-domain incompatibility, with
internal teacher saturation observed on this input. Do not conclude that every
RouteNet-Fermi checkpoint fails on realistic traffic, or that our offline results
are invalid. Do not claim a new KD routing improvement, real-time guarantee,
energy saving, physical edge deployment, or a repaired Abilene model.

### Evidence access and reporting discipline

Use the artifact paths in Section 5. The authoritative repeated-trial calculation
is `outputs/six_validation_summary.json`; it records raw result filenames/hashes,
checkpoint identities and limitations. Frozen predictions/timings are in
`outputs/six_validation_predictions.json`. Author comparison source provenance is
in `outputs/author_real_traffic/provenance.json`, pinned to repository commit
`ef72a37bb6f677e149f354f088f5836f29c862b5`; results are in `comparison.json`.

An agent with GitHub-only access will not have these ignored artifacts. It can
use this README as the documented summary, but must not claim independent raw-
data verification. Obtain the artifacts if checking/recalculating results is
required. Retain the negative findings and provenance; never invent missing
measurements, statistical significance, graphs or additional completed work.
