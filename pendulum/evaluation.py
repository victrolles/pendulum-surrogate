from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .inference import infer
    from .simulation import simulate
else:
    from inference import infer
    from simulation import simulate


def evaluate(
        model_path: str,
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        num_steps: int = 2000,
        l: float = 1.0,
        m: float = 1.0,
        b: float = 0.1,
        theta0: float = np.pi / 4,
        omega0: float = 0.0,
        t: np.ndarray | None = None,
        show: bool = True
) -> dict[str, float]:
    """Compare predictions and simulation at the same ``num_steps`` instants.

    ``t`` includes the initial instant and must contain ``num_steps + 1``
    uniformly spaced times. By default, the time step is 0.02 seconds.
    The time step must match the one used to train the model: the network
    predicts one transition per call and cannot change its physical time step.

    Simulation timing covers the simulate call; inference timing covers only
    the prediction loop, excluding loading, setup, warm-up, and result transfers.
    Plotting and error calculations are excluded from both timings.
    Return timings in seconds, simulation/inference time ratio, MSE, and mean
    absolute errors (MAE): mean(abs(prediction - reference)). Errors are in rad
    for theta and rad/s for omega, without normalization or percentages.
    """
    if not isinstance(num_steps, (int, np.integer)) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")

    if t is None:
        t = np.arange(num_steps + 1, dtype=float) * 0.02
    else:
        t = np.asarray(t, dtype=float)

    if t.shape != (num_steps + 1,) or not np.isfinite(t).all():
        raise ValueError("t must contain num_steps + 1 finite times, including the initial instant.")
    time_steps = np.diff(t)
    if np.any(time_steps <= 0) or not np.allclose(time_steps, time_steps[0], rtol=1e-6, atol=0):
        raise ValueError("t must be strictly increasing and uniformly spaced.")

    start = perf_counter()
    theta_sim, omega_sim = simulate(l=l, m=m, b=b, theta0=theta0, omega0=omega0, t=t)
    simulation_time = perf_counter() - start

    theta_infer, omega_infer, inference_time = infer(
        model_path=model_path,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_steps=num_steps,
        l=l,
        m=m,
        b=b,
        theta0=theta0,
        omega0=omega0,
        return_timing=True
    )

    # Predictions are the states after each transition; skip the simulated initial state.
    theta_sim = np.asarray(theta_sim, dtype=float)[1:]
    omega_sim = np.asarray(omega_sim, dtype=float)[1:]
    theta_infer = np.asarray(theta_infer, dtype=float)
    omega_infer = np.asarray(omega_infer, dtype=float)
    trajectories = (theta_sim, omega_sim, theta_infer, omega_infer)
    if any(values.shape != (num_steps,) for values in trajectories):
        raise ValueError("Simulation and inference must each return num_steps comparable states.")
    if any(not np.isfinite(values).all() for values in trajectories):
        raise ValueError("Simulation or inference produced non-finite values; the trajectory may have diverged.")

    metrics = {
        "inference_time": inference_time,
        "simulation_time": simulation_time,
        "speedup": simulation_time / inference_time if inference_time > 0 else float("inf"),
        "mse_theta": float(np.mean((theta_infer - theta_sim) ** 2)),
        "mse_omega": float(np.mean((omega_infer - omega_sim) ** 2)),
        "mae_theta": float(np.mean(np.abs(theta_infer - theta_sim))),
        "mae_omega": float(np.mean(np.abs(omega_infer - omega_sim)))
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(right=0.72, hspace=0.35)
    fig.suptitle("Pendulum: inference vs simulation")
    for ax, prediction, reference, label in (
        (axes[0], theta_infer, theta_sim, r"$\theta$ (rad)"),
        (axes[1], omega_infer, omega_sim, r"$\omega$ (rad/s)")
    ):
        ax.plot(t[1:], prediction, color="tab:blue", linewidth=2, label="Inference")
        ax.plot(t[1:], reference, color="tab:orange", linestyle="--", linewidth=1.2, label="Simulation")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title("Angle (rad)")
    axes[1].set_title("Vitesse angulaire (rad/s)")
    axes[1].set_xlabel("Time (s)")

    stats = (
        f"Compared points: {num_steps:,}\n"
        f"Time step: {time_steps[0]:.3f} s\n\n"
        f"Inference: {inference_time:.3f} s\n"
        f"Simulation: {simulation_time:.3f} s\n\n"
        f"Ratio simulation / inference:\n{metrics['speedup']:.3f}x\n"
        "(> 1: inference is faster)\n\n"
        f"Erreur theta: {metrics['mae_theta']:.4g} rad\n"
        f"Erreur omega: {metrics['mae_omega']:.4g} rad/s\n\n"
        "Timing: inference loop only\n(model loading excluded)."
    )
    fig.text(0.75, 0.85, stats, va="top", fontsize=10, linespacing=1.5,
             bbox={"boxstyle": "round,pad=0.8", "facecolor": "whitesmoke", "edgecolor": "lightgray"})

    if show:
        plt.show()
    return metrics
