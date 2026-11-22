"""T4 offline six-switch diagnostic; no Mininet, sudo, servers or training.

Uses the unchanged model_worker adapters and physical guards. Measured trial
means are used only to score predictions, never as model inputs. Outputs must
be reviewed at R3 before their conclusions are added to SUMMARY.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def measurements(path, scenario_hash, prediction_path):
    if not path.exists():
        means = [123.785, 112.929, 196.331]
        return dict(source="report_table_fallback", source_sha256=None,
                    means_ms=means, trial_means_ms=[[m] for m in means],
                    note="Only rounded candidate means available; not a 15-trial APE.")
    summary = json.loads(path.read_text())
    if summary["scenario_sha256"] != scenario_hash:
        raise ValueError("Measured summary refers to a different scenario")
    if prediction_path.exists() and summary.get("prediction_source_sha256"):
        if summary["prediction_source_sha256"] != sha256(prediction_path):
            raise ValueError("Measured summary prediction-source hash does not match")
    candidates = {row["candidate"]: row for row in summary["candidates"]}
    if set(candidates) != {0, 1, 2}:
        raise ValueError("Expected three measured candidates")
    means = [float(candidates[k]["mean_delay_ms"]) for k in range(3)]
    trials = [[float(v) for v in candidates[k]["trial_mean_delay_ms"]] for k in range(3)]
    if any(v <= 0 for v in means) or any(not row or any(v <= 0 for v in row) for row in trials):
        raise ValueError("APE requires positive measured delays")
    return dict(source=str(path), source_sha256=sha256(path), means_ms=means, trial_means_ms=trials,
                note="APE is against foreground measurements; background flows have no measured labels here.")


def run(args, document):
    # Importing model_worker sets its historical one-thread/CPU environment
    # before TensorFlow starts. Importing it does not start a server or Mininet.
    from model_worker import scenario, build_inputs, load_model, prediction, physical_checks
    from settings import CHECKPOINTS, digest
    import numpy as np
    import tensorflow as tf

    config = scenario()
    scenario_hash = digest(config)
    prediction_path = HERE / "outputs/six_validation_predictions.json"
    frozen = json.loads(prediction_path.read_text()) if prediction_path.exists() else None
    scenario_source = "settings.scenario"
    if frozen is not None:
        if digest(frozen["scenario"]) != frozen["scenario_sha256"]:
            raise ValueError("Frozen scenario hash does not match its contents")
        # The measured validation added measurement_s=60.0 to the pilot.
        # That field controls trial duration and is never read by build_inputs.
        # Compare every other field, then retain the exact frozen provenance.
        current_inputs = {k: v for k, v in config.items() if k != "measurement_s"}
        frozen_inputs = {k: v for k, v in frozen["scenario"].items() if k != "measurement_s"}
        if digest(current_inputs) != digest(frozen_inputs):
            raise ValueError("Current and frozen model-input scenarios differ")
        if scenario_hash != frozen["scenario_sha256"]:
            print("Scenario input fields match; retaining frozen measurement_s metadata.", flush=True)
        config = frozen["scenario"]
        scenario_hash = frozen["scenario_sha256"]
        scenario_source = "frozen_scenario_verified_against_settings_except_measurement_duration"
    measured = measurements(HERE / "outputs/six_validation_summary.json", scenario_hash, prediction_path)
    measured_ranking = sorted(range(3), key=lambda k: (measured["means_ms"][k], k))
    document.update(scenario=config, scenario_sha256=scenario_hash, scenario_source=scenario_source, measured=measured,
                    measured_ranking=measured_ranking,
                    frozen_source=str(prediction_path) if frozen else None,
                    frozen_source_sha256=sha256(prediction_path) if frozen else None,
                    frozen_tolerance=dict(rtol=1e-6, atol_seconds=1e-8),
                    tensorflow=tf.__version__, models={})
    save(args.out, document)

    for name in ("teacher", "scratch", "kd"):
        checkpoint = ROOT / CHECKPOINTS[name]
        files = [Path(str(checkpoint) + ".index")] + sorted(checkpoint.parent.glob(checkpoint.name + ".data-*"))
        if len(files) < 2:
            raise FileNotFoundError(str(checkpoint))
        hashes = {p.name: sha256(p) for p in files}
        expected = frozen["models"][name] if frozen else None
        if expected is not None and (expected["checkpoint"] != CHECKPOINTS[name] or expected["checkpoint_sha256"] != hashes):
            raise ValueError("Checkpoint differs from frozen source: " + name)
        print("Loading", name, CHECKPOINTS[name], flush=True)
        model = load_model(name, config)
        result = dict(checkpoint=CHECKPOINTS[name], checkpoint_sha256=hashes, orders={})
        document["models"][name] = result
        for order, preserve in (("hop", True), ("ref", False)):
            candidates = []
            for k in range(3):
                inputs = build_inputs(config, k, preserve_hop_order=preserve)
                foreground_s = float(prediction(model, inputs)[config["foreground"]])
                physical = physical_checks(model, inputs)
                predicted_ms = foreground_s * 1000
                observed_ms = measured["means_ms"][k]
                apes = [100 * abs(predicted_ms - v) / v for v in measured["trial_means_ms"][k]]
                candidates.append(dict(candidate=k, path=config["candidates"][k],
                                       predicted_foreground_delay_s=foreground_s,
                                       predicted_foreground_delay_ms=predicted_ms,
                                       measured_mean_delay_ms=observed_ms,
                                       ape_vs_candidate_mean_percent=100 * abs(predicted_ms - observed_ms) / observed_ms,
                                       ape_vs_trial_means_percent=apes, physical_checks=physical))
            ranking = sorted(range(3), key=lambda k: (candidates[k]["predicted_foreground_delay_ms"], k))
            valid = all(c["physical_checks"]["valid"] for c in candidates)
            # The worker's controller interface permits a choice only when all
            # candidate physical guards pass. Retain the raw ranking regardless.
            selected = ranking[0] if valid else None
            trial_apes = [v for c in candidates for v in c["ape_vs_trial_means_percent"]]
            scored = dict(preserve_hop_order=preserve, candidates=candidates,
                          predicted_ranking=ranking, argmin_candidate=ranking[0],
                          physical_checks_all_valid=valid, selected_candidate=selected,
                          ranking_matches_measured=ranking == measured_ranking,
                          ranking_1_0_2_preserved=ranking == [1, 0, 2], choice_candidate1_preserved=selected == 1,
                          mean_ape_across_candidate_means_percent=float(np.mean([
                              c["ape_vs_candidate_mean_percent"] for c in candidates])),
                          n_foreground_trial_means=len(trial_apes),
                          mean_ape_across_available_foreground_trial_means_percent=float(np.mean(trial_apes)))
            result["orders"][order] = scored
            if preserve:
                if expected is None:
                    result["frozen_check"] = dict(status="unavailable", note="Frozen prediction file absent")
                else:
                    actual = np.array([c["predicted_foreground_delay_s"] for c in candidates])
                    reference = np.array(expected["predicted_foreground_delay_s"])
                    passed = bool(np.allclose(actual, reference, rtol=1e-6, atol=1e-8))
                    result["frozen_check"] = dict(status="passed" if passed else "failed",
                                                  max_absolute_difference_ms=float(np.max(np.abs(actual-reference))*1000),
                                                  selected_candidate_matches=selected == expected["selected_candidate"])
                    save(args.out, document)
                    if not passed or selected != expected["selected_candidate"]:
                        raise ValueError("True-hop predictions do not reproduce frozen source: " + name)
            save(args.out, document)
            print(name, order, "foreground ms=", [round(c["predicted_foreground_delay_ms"], 6) for c in candidates],
                  "ranking=", ranking, "selected=", selected, "physical_valid=", valid,
                  "trial-mean APE%=", round(scored["mean_ape_across_available_foreground_trial_means_percent"], 6), flush=True)
        del model
        tf.keras.backend.clear_session()

    document.update(status="complete", finished_unix=time.time(), review_status="pending_R3",
                    training_order_ranking_survives_for_all_models=all(
                        m["orders"]["ref"]["ranking_1_0_2_preserved"] for m in document["models"].values()),
                    training_order_choice_survives_for_all_models=all(
                        m["orders"]["ref"]["choice_candidate1_preserved"] for m in document["models"].values()),
                    note="Diagnostic only. No Mininet rerun or new measured network outcomes. "
                         "Review at R3 before adding conclusions to SUMMARY.md.")
    save(args.out, document)
    print("Saved", args.out, "— pending R3 review", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/dec1/emulation_hop_order.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new path to preserve the previous diagnostic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    document = dict(schema_version=1, status="running", started_unix=time.time())
    try:
        document["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        document["git_commit"] = None
    try:
        run(args, document)
    except Exception as error:
        document.update(status="failed", error="{}: {}".format(type(error).__name__, error))
        save(args.out, document)
        summary = ROOT / "results/dec1/SUMMARY.md"
        with summary.open("a") as out:
            out.write("\n## Problems\n\nT4 stopped: {}. See {}.\n".format(document["error"], args.out))
        raise


if __name__ == "__main__":
    main()
