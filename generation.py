import json
from collections.abc import Sequence
from enum import Enum
from itertools import product
from math import isclose, prod
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import scipy
from tqdm import tqdm

from simulation import simulate


class Method(str, Enum):
    fixed_sampling = "fixed_sampling"
    random_sampling = "random_sampling"


def _time_grid(duration: float, framerate: float) -> np.ndarray:
    if not np.isfinite(duration) or not np.isfinite(framerate) or duration <= 0 or framerate <= 0:
        raise ValueError("duration and framerate must be finite and positive.")
    steps = duration * framerate
    num_steps = round(steps) if np.isfinite(steps) else 0
    if num_steps < 1 or not isclose(steps, num_steps, abs_tol=1e-9, rel_tol=0):
        raise ValueError("duration × framerate must give a positive whole number of transitions.")
    return np.linspace(0, duration, num_steps + 1)


def generate_dataset(
    dataset_dir: str | Path = "datasets",
    method: Method | str = Method.fixed_sampling,
    num_samples: int = 1000,
    range_l: Sequence[float] = (1.0, 2.0),
    range_m: Sequence[float] = (1.0, 2.0),
    range_b: Sequence[float] = (0.1, 0.2),
    range_theta0: Sequence[float] = (0.0, np.pi / 2),
    range_omega0: Sequence[float] = (0.0, 1.0),
    duration: float = 10.0,
    framerate: float = 50.0,
    *,
    seed: int | None = None,
    t: np.ndarray | None = None,
) -> Path:
    """Save physical parameters, state trajectories and JSON metadata in an NPZ.

    Fixed sampling uses the Cartesian product of the supplied values. Random
    sampling draws num_samples trajectories independently and uniformly from
    each [min, max] range. A single value fixes that parameter in either mode.
    A random seed is recorded even when the caller leaves it unspecified.
    When provided, the uniform time grid t overrides duration and framerate.
    """
    method = Method(method)
    ranges = {
        "l": np.asarray(range_l, dtype=float),
        "m": np.asarray(range_m, dtype=float),
        "b": np.asarray(range_b, dtype=float),
        "theta0": np.asarray(range_theta0, dtype=float),
        "omega0": np.asarray(range_omega0, dtype=float),
    }
    for name, values in ranges.items():
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError(f"range_{name} must contain one or more finite values.")
    if np.any(ranges["l"] <= 0) or np.any(ranges["m"] <= 0):
        raise ValueError("Pendulum length and mass must be positive.")

    if method is Method.fixed_sampling:
        if seed is not None:
            raise ValueError("seed is only used for random_sampling.")
        num_samples = prod(len(values) for values in ranges.values())
        combinations = product(*ranges.values())
    else:
        if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer for random_sampling.")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer.")
        num_samples = int(num_samples)
        for name, values in ranges.items():
            if values.size > 2 or values[0] > values[-1]:
                raise ValueError(f"range_{name} must be a fixed value or [min, max] for random_sampling.")
        lows = [values[0] for values in ranges.values()]
        highs = [values[-1] for values in ranges.values()]
        seed_sequence = np.random.SeedSequence(seed)
        seed = int(seed_sequence.entropy)
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        combinations = rng.uniform(lows, highs, size=(num_samples, len(ranges)))

    if t is None:
        t = _time_grid(duration, framerate)
    else:
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2 or not np.isfinite(t).all():
            raise ValueError("t must contain at least two finite times.")
        steps = np.diff(t)
        if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
            raise ValueError("t must be strictly increasing and uniformly spaced.")

    num_steps = len(t) - 1
    duration = float(t[-1] - t[0])
    time_step = duration / num_steps
    parameters = np.empty((num_samples, 3))
    states = np.empty((num_samples, len(t), 2))
    for index, (l, m, b, theta0, omega0) in enumerate(tqdm(combinations, total=num_samples)):
        theta, omega = simulate(l, m, b, theta0, omega0, t)
        if len(theta) != len(t) or len(omega) != len(t):
            raise ValueError("Simulation did not return a state at every requested time.")
        if not np.isfinite(theta).all() or not np.isfinite(omega).all():
            raise ValueError("Simulation returned a non-finite state.")
        parameters[index] = l, m, b
        states[index, :, 0] = theta
        states[index, :, 1] = omega

    num_transitions = num_samples * num_steps
    metadata = {
        "schema_version": 2,
        "method": method.value,
        "seed": seed,
        "random_generator": "PCG64" if seed is not None else None,
        "num_samples": num_samples,
        "ranges": {name: values.tolist() for name, values in ranges.items()},
        "start_time": float(t[0]),
        "duration": duration,
        "framerate": 1 / time_step,
        "time_step": time_step,
        "num_points": len(t),
        "num_transitions": num_transitions,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / f"pendulum_dataset_{num_transitions}_{uuid4().hex}.npz"
    with TemporaryDirectory(dir=dataset_dir) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        np.savez_compressed(
            temp_path, parameters=parameters, states=states, metadata=json.dumps(metadata, allow_nan=False)
        )
        temp_path.replace(output_path)
    print(f"Dataset saved to {output_path}")
    return output_path
