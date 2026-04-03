import os
import numpy as np
from datanetAPI import DatanetAPI

def count_root(root, name):
    raw = 0
    usable = 0
    filtered = 0
    delays = []

    for scenario in sorted(os.listdir(root)):
        scenario_path = os.path.join(root, scenario)
        if not os.path.isdir(scenario_path):
            continue

        try:
            tool = DatanetAPI(scenario_path, shuffle=False)
        except Exception as e:
            print(f"Skipping {scenario_path}: {e}")
            continue

        for s in tool:   # s is a Sample object
            raw += 1
            try:
                y = float(s.get_global_delay())
            except Exception:
                filtered += 1
                continue

            if not (y > 0):
                filtered += 1
                continue

            usable += 1
            delays.append(y)

    delays = np.array(delays, dtype=np.float64)

    print(f"\n{name}: {root}")
    print(f"  Raw samples seen        : {raw}")
    print(f"  Filtered (non-positive) : {filtered}")
    print(f"  Usable yielded          : {usable}")
    print(f"  Usable ratio            : {(usable/raw) if raw else 0:.3f}")

    if len(delays) > 0:
        print("  Delay stats (global_delay):")
        print(f"    min    : {delays.min():.6g}")
        print(f"    max    : {delays.max():.6g}")
        print(f"    mean   : {delays.mean():.6g}")
        print(f"    median : {np.median(delays):.6g}")
        for th in [1e-4, 1e-3]:
            pct = (delays < th).mean() * 100
            print(f"    % < {th:g}: {pct:.3f}%")

if __name__ == "__main__":
    count_root("../data/all_mixed/train_teacher", "TRAIN_TEACHER")
    count_root("../data/all_mixed/val_teacher",   "VAL_TEACHER")
    count_root("../data/all_mixed/test",          "TEST")
