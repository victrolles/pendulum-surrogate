from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from .inference import infer
from .simulation import _simulate_on_grid, time_grid


def evaluate(
        model_path: str | Path,
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        l: float = 1.0,
        m: float = 1.0,
        b: float = 0.1,
        theta0: float = np.pi / 4,
        omega0: float = 0.0,
        duration: float = 10.0,
        framerate: float = 50.0,
        show: bool = True,
        *,
        start_time: float = 0.0,
        relative_error_floor: float = 0.1,
        plot_path: str | Path | None = None,
) -> dict[str, float]:
    """Compare predictions and simulation over duration seconds at framerate Hz.

    The model predicts one transition per call, so framerate must match the
    dataset used to train it. The initial state is shared and excluded from
    the comparison.

    Simulation timing covers the solver call; inference timing covers only
    the prediction loop, excluding loading, setup, warm-up, and result transfers.
    Plotting and error calculations are excluded from both timings.
    Return timings in seconds, simulation/inference time ratio, MSE, and mean
    absolute errors (MAE) in rad and rad/s. Relative errors are percentages
    computed with max(abs(reference), relative_error_floor) as the denominator
    in each channel. If plot_path is set, save the comparison figure there.
    """
    if not np.isfinite(relative_error_floor) or relative_error_floor <= 0:
        raise ValueError("relative_error_floor must be finite and positive.")

    t = time_grid(duration, framerate, start_time=start_time)
    num_steps = len(t) - 1

    start = perf_counter()
    theta_sim, omega_sim = _simulate_on_grid(l, m, b, theta0, omega0, t)
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

    theta_error = np.abs(theta_infer - theta_sim)
    omega_error = np.abs(omega_infer - omega_sim)
    theta_relative_pct = float(100 * np.mean(theta_error / np.maximum(np.abs(theta_sim), relative_error_floor)))
    omega_relative_pct = float(100 * np.mean(omega_error / np.maximum(np.abs(omega_sim), relative_error_floor)))
    metrics = {
        "inference_time": inference_time,
        "simulation_time": simulation_time,
        "speedup": simulation_time / inference_time if inference_time > 0 else float("inf"),
        "mse_theta": float(np.mean(theta_error ** 2)),
        "mse_omega": float(np.mean(omega_error ** 2)),
        "mae_theta": float(np.mean(theta_error)),
        "mae_omega": float(np.mean(omega_error)),
        "theta_relative_error_pct": theta_relative_pct,
        "omega_relative_error_pct": omega_relative_pct,
    }

    if not show and plot_path is None:
        return metrics

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(right=0.72, hspace=0.35)
    fig.suptitle("Pendulum: inference vs simulation")
    for ax, prediction, reference, label in (
        (axes[0], theta_infer, theta_sim, r"$\theta$ (rad)"),
        (axes[1], omega_infer, omega_sim, r"$\omega$ (rad/s)")
    ):
        ax.plot(t[1:], prediction, color="tab:blue", linestyle="-", linewidth=2, label="Inference")
        ax.plot(t[1:], reference, color="tab:orange", linestyle="--", linewidth=1.2, label="Simulation")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title("Angle (rad)")
    axes[1].set_title("Vitesse angulaire (rad/s)")
    axes[1].set_xlabel("Time (s)")

    stats = (
        f"Compared points: {num_steps:,}\n"
        f"Time step: {1 / framerate:.3f} s\n\n"
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

    if plot_path is not None:
        fig.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return metrics
