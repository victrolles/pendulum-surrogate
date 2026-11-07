# surrogate-models

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
python pendulum/runner.py --mode simulate
python pendulum/runner.py --mode infer
python pendulum/runner.py --mode train
```

## Angular acceleration experiment

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
