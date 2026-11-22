
## Problems

T3 benchmark stopped: DatanetException: ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]. See results/dec1/latency_v2_runs_20261122_205401/baseline_default.json.

Resolution (22 Nov 2026): the bundled `DatanetAPI.set_files_to_process` shadows the built-in `tuple`, so its type check rejects valid selections. T3 now copies the selected tuples directly into the API's `_selected_tuple_files` field, preserving the original archive order. Historical API code and checkpoints were not modified. Read-only checks for 200, 240, 260, 280 and 300 nodes all matched T2's first scenario IDs; see `latency_v2_selector_check.log`. No model inference or full benchmark restart was performed by the execution agent. The user subsequently restarted the queue in `latency_v2_runs_20261122_212858/`; it completed all 48 combinations. The failed attempt is preserved.

T3 historical-timing deviation (22 Nov 2026): teacher/default/validation measured **181.6339 ms**, +1.76% versus 178.5 ms (within 20%). Scratch/default/validation measured **41.1798 ms**, +20.41% versus 34.2 ms, slightly outside the 20% bound of 41.04 ms. The cause has not been established. All eight workers and 48 combinations completed successfully; use `latency_v2.json` for the new measurements and report this deviation rather than claiming both reproduction checks passed. No rerun was started to chase the historical value. Full coverage and completion were checked; T4 has not started.
