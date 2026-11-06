from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pendulum import simulate

def generator():

    # 1. Range of initial conditions and parameters for the pendulum simulation
    list_l = [1.0]  # length of the pendulum (m)
    list_m = [1.0]  # mass of the pendulum (kg)
    list_b = [0.1]  # damping coefficient (kg/s)
    list_theta0 = [np.pi / 4]  # initial angles (radians)
    list_omega0 = [0.0]  # initial angular velocities (rad/s)
    t = np.linspace(0, 10, 500)

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
    output_path = Path(f"datasets/pendulum/pendulum_dataset_{dataset.shape[0]}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, dataset=dataset)



if __name__ == "__main__":

    # Generate the dataset
    generator()

    # Load the dataset
    data = np.load("datasets/pendulum/pendulum_dataset_499.npz")

    # Display the first 5 rows
    print(data["dataset"][:5])
