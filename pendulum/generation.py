from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from .simulation import simulate
else:
    from simulation import simulate

def generate_dataset(
        dataset_dir: str = "pendulum/datasets",  # directory to save the dataset
        list_l: list[float] = [1.0, 2.0],        # length of the pendulum (m)
        list_m: list[float] = [1.0, 2.0],        # mass of the pendulum (kg)
        list_b: list[float] = [0.1, 0.2],        # damping coefficient (kg/s)
        list_theta0: list[float] = [np.pi / 4],  # initial angles (radians)
        list_omega0: list[float] = [0.0],        # initial angular velocities (rad/s)
        t: np.ndarray = np.linspace(0, 10, 500)  # time array (s)
) -> Path:

    # 2. Generate all parameter combinations
    combinations = list(product(list_l, list_m, list_b, list_theta0, list_omega0))

    print(f"Total combinations: {len(combinations)}")

    # 3. Iterate through combinations
    dataset = []

    for l, m, b, theta0, omega0 in tqdm(combinations):
        # 4. Simulate the pendulum for each combination
        theta, omega = simulate(l, m, b, theta0, omega0, t)

        # Create one array containing all transitions
        trajectory = np.column_stack([
            np.full(len(t) - 1, l),
            np.full(len(t) - 1, m),
            np.full(len(t) - 1, b),
            theta[:-1],
            omega[:-1],
            theta[1:],
            omega[1:]
        ])

        dataset.append(trajectory)

    # Combine every trajectory into one dataset
    dataset = np.vstack(dataset)

    # Save the dataset
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    output_path = dataset_dir / f"pendulum_dataset_{dataset.shape[0]}.npz"
    np.savez_compressed(output_path, dataset=dataset)
    print(f"Dataset saved to {output_path}")
    return output_path
