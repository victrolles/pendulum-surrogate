from math import isclose

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81  # acceleration due to gravity (m/s^2)


def time_grid(duration: float, framerate: float, *, start_time: float = 0.0) -> np.ndarray:
    """Build a time grid including the initial state and the final state."""
    if not np.isfinite(duration) or not np.isfinite(framerate) or duration <= 0 or framerate <= 0:
        raise ValueError("duration and framerate must be finite and positive.")
    if not np.isfinite(start_time) or not np.isfinite(start_time + duration):
        raise ValueError("The start and end times must be finite.")
    steps = duration * framerate
    num_steps = round(steps) if np.isfinite(steps) else 0
    if num_steps < 1 or not isclose(steps, num_steps, abs_tol=1e-9, rel_tol=0):
        raise ValueError("duration × framerate must give a positive whole number of transitions.")
    t = np.linspace(start_time, start_time + duration, num_steps + 1)
    if start_time and not np.allclose(np.diff(t), duration / num_steps, rtol=1e-6, atol=0):
        raise ValueError("The time step is too small for the requested start time.")
    return t


def simulate(
        l: float = 1.0,       # length of the pendulum (m)
        m: float = 1.0,       # mass of the pendulum (kg)
        b: float = 0.2,       # damping coefficient (kg/s)
        theta0: float = 0.0,  # initial angle (radians)
        omega0: float = 0.0,  # initial angular velocity (rad/s)
        duration: float = 10.0,  # simulation duration (s)
        framerate: float = 50.0  # samples per second (Hz)
    ) -> tuple[np.ndarray, np.ndarray]:
    """Return angle and angular velocity on the requested time grid."""
    return _simulate_on_grid(l, m, b, theta0, omega0, time_grid(duration, framerate))


def _simulate_on_grid(
    l: float, m: float, b: float, theta0: float, omega0: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Physical equation of the pendulum
    def pendulum(t, y):
        theta, omega = y
        dtheta = omega
        domega = -(G / l) * np.sin(theta) - (b / m * l * l) * omega
        return [dtheta, domega]

    # 2. Solve the differential equations at the requested times
    solution = solve_ivp(
        pendulum,
        [t[0], t[-1]],
        [theta0, omega0],
        t_eval=t
    )

    return solution.y[0], solution.y[1]
