# Periapsis and black hole absorption

Numerical archive for *How the Direction of Periapsis Controls Black Hole Absorption from a Boson Cloud*. The Letter compares two initially empty absorbing modes on orbits starting at arguments of periapsis of 93 and 150 degrees. Both orbits migrate from 3.25 to 3.10 gravitational Bohr radii. The horizon count is a fraction of the initial parent population.

The code and numerical data are distributed under the [MIT License](LICENSE).

## Reading the results

- [Migration summaries](data/selection_migrations.json) contain the initial conditions, endpoint counts and interference comparisons. In the run names, `3` and `60` are offsets from 90 degrees, `C` retains angular interference, `D` adds angular powers separately, and `zero` denotes an initially empty mode. The Letter compares two `C-zero` histories.
- [Trajectories](data/selection_traces.npz) contain named columns for 15 histories. [Additional samples](data/selection_samples.json.gz) contain the saved complex mode amplitudes and orbital diagnostics for all 37 runs. For densely sampled runs, shared columns are stored in the NPZ and the two tables can be joined on time `s`.
- [Cloud and absorbing mode](data/formed_periapsis.json), [formation rates](data/formation_rates.json) and [formation stages](data/selection_formation.json) describe the background. [Fixed orbits](data/selection_orbits.json) give the libration areas and intrinsic horizon counts; [moving axes](data/moving_axes.json) contains the reference evolutions.
- The continuum inputs are [ionization projections](data/ionization_projections.json.gz) and response grids for [the hole alone](data/selection_grid.npz) or [with the spherical cloud potential](data/selection_monopole.npz). The [positive-harmonic amplitudes](data/selection_open_vertices.npz), [horizon estimates](data/selection_open_horizon.json) and [resolution comparisons](data/selection_accuracy.json) accompany these calculations. The [93-degree](data/selection_shape_93.json) and [150-degree](data/selection_shape_150.json) histories include quadrupolar cloud deformation.

## Using the code

From the repository root with Python 3.11:

```sh
python -m pip install -r requirements.txt
python scripts/horizon_counts.py
python scripts/migration_figure.py
```

These two scripts read the archive. To evolve the physical pair anew, run `python scripts/selection_migration.py migrate --angles 3 60 --models C --zero --trace`. Add `--monopole` for the spherical continuum, or `--shape --fine` for the deformation comparison. Individual migrations write to `runs/selection/`.
