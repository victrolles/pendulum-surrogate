from time import perf_counter

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .model import PendulumModel

def infer(
        model_path: str,            # path to the trained model
        hidden_size: int = 64,      # size of the hidden layers
        num_hidden_layers: int = 1, # number of hidden layers
        num_steps: int = 1000,      # number of inference steps
        l: float = 1.0,             # length of the pendulum (m)
        m: float = 1.0,             # mass of the pendulum (kg)
        b: float = 0.1,             # damping coefficient (kg/s)
        theta0: float = np.pi / 4,  # initial angle (radians)
        omega0: float = 0.0,        # initial angular velocity (rad/s)
        *,
        return_timing: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
    """Predict successive states, optionally returning the inference loop time.

    With return_timing=True, return (theta, omega, elapsed_seconds). The timer
    covers predictions, recurrent state updates, and storing states on device.
    Loading, initial tensor allocation, warm-up, progress display, and transfer
    of the completed trajectory to NumPy are excluded. GPU work is synchronized.
    The default return value remains (theta, omega).
    """

    # Set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Generate simulation start state
    input = torch.tensor(
        [[l, m, b, theta0, omega0]],
        dtype=torch.float32,
        device=device
    )

    # Define the model
    model = PendulumModel(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers
    )

    # Load the trained model weights
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to inference mode

    trajectory = torch.empty((num_steps, 2), dtype=torch.float32, device=device)
    steps = range(num_steps) if return_timing else tqdm(range(num_steps))

    with torch.inference_mode():
        if return_timing:
            # Initialize the execution backend without advancing the initial state.
            model(input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        for step in steps:
            output = model(input)
            trajectory[step] = output[0]
            input = torch.cat([input[:, :3], output], dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_time = perf_counter() - start

    # Convert trajectory to a NumPy array
    trajectory = trajectory.cpu().numpy()
    theta = trajectory[:, 0]
    omega = trajectory[:, 1]

    if return_timing:
        return theta, omega, inference_time
    return theta, omega
