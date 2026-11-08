# pendulum-surrogate

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
python runner.py --mode simulate
python runner.py --mode infer
python runner.py --mode train
```

## Pendulum datasets

```bash
python generate_dataset.py --yaml configs/dataset.yaml
python runner.py --mode train --dataset_path PATH_TO_GENERATED_NPZ
```

Copy the example YAML to define a new dataset. Paths in `dataset_dir` are
relative to the YAML file and default to `datasets` alongside it. In
`random_sampling`, set `num_samples` to the number of trajectories and
provide an integer `seed`. Each parameter accepts a constant or a
`[min, max]` interval. In `fixed_sampling`, list the values to combine in a
Cartesian grid. The grid determines `num_samples` and you can specify it
as a check. Use seconds for `duration`, Hz for `framerate`, and radians
and rad/s for `theta0` and `omega0`.

Each archive stores `[l, m, b]` once per trajectory and a series of
`[theta, omega]` states including the initial state. JSON metadata records
the sampling settings, seed and NumPy/SciPy versions. Repeat a run with the
same YAML, simulator and environment to reproduce the arrays. The
`PendulumDataset` reader creates input and target pairs by indexing
consecutive states and also reads legacy seven-column archives.

## Angular acceleration experiment on a separate branch

Branch `experiment/pendulum-acceleration` predicts three outputs per step:
angle theta (rad), angular velocity omega (rad/s), and angular acceleration
alpha = d(omega)/dt (rad/s²).

`solve_ivp` still integrates only theta and omega. Acceleration is evaluated from
the same equation at every simulated state, without numerical differentiation.
The model uses six inputs `[l, m, b, theta, omega, alpha]` and learns the targets
`[theta_next, omega_next, alpha_next]`. Initial acceleration is computed from
the physical equation. Each inference step feeds all three predictions back
into the next input, including the predicted acceleration. This experiment does not
guarantee better angle or velocity predictions; the training loss averages
squared errors across all three outputs, which have different units and scales.

Datasets use nine columns:
`[l, m, b, theta, omega, alpha, theta_next, omega_next, alpha_next]`.
They are saved in `pendulum/datasets/acceleration`, and new models in
`pendulum/models/acceleration`. Existing datasets with seven or eight columns
must be regenerated, and models with five inputs must be retrained.
The dataset and checkpoint record the training time step; inference and
evaluation reject a different requested step.

```bash
python -m pendulum.runner --mode train
python -m pendulum.runner --mode infer
python -m pendulum.runner --mode evaluate
```

The defaults train 99 transitions over 10 seconds and save `model_99.pth`.
For custom runs, use `duration / num_steps` in inference and evaluation equal
to `duration / (num_points - 1)` during training. All three trajectories are
plotted, with MAE in rad, rad/s, and rad/s². Inference timings continue to exclude
loading, setup, warm-up, and result transfers.
