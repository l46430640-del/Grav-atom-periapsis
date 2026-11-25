"""Read the two physical trajectories and report the horizon count comparison."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson


DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    grid = json.loads((DATA / "formed_periapsis.json").read_bytes())["grid"]
    gamma = grid["pole"]["population_width"]
    with np.load(DATA / "selection_traces.npz") as archive:
        for potential in ("", "-monopole"):
            counts, times = [], []
            for offset in (3, 60):
                name = f"migration-{offset}-0.0001-C{potential}-zero"
                row = dict(zip(archive["columns"], archive[name].T))
                tau = row["s"] / grid["parameters"]["cloud"]
                count = gamma * simpson(row["excitation"], x=tau) + row["negative_horizon_fraction"][-1]
                print(f"{name}: F_H = {count:.4g}")
                counts.append(count)
                times.append(tau[-1])
            count_ratio = counts[1] / counts[0]
            time_ratio = times[1] / times[0]
            print(f"Count ratio {count_ratio:.3g}, time ratio {time_ratio:.3g}, mean rate ratio {count_ratio/time_ratio:.3g}")

    shape = [json.loads((DATA / f"selection_shape_{angle}.json").read_bytes()) for angle in (93, 150)]
    print(f"Count ratio with cloud deformation: {shape[1]['horizon_fraction'] / shape[0]['horizon_fraction']:.3g}")


if __name__ == "__main__":
    main()
