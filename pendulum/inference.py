import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

if __package__:
    from .model import PendulumModel
else:
    from model import PendulumModel

def infer(
        model_path: str,            # path to the trained model
        hidden_size: int = 64,      # size of the hidden layers
        num_hidden_layers: int = 1, # number of hidden layers
        num_steps: int = 1000,      # number of inference steps
        l: float = 1.0,             # length of the pendulum (m)
        m: float = 1.0,             # mass of the pendulum (kg)
        b: float = 0.1,             # damping coefficient (kg/s)
        theta0: float = np.pi / 4,  # initial angle (radians)
        omega0: float = 0.0):       # initial angular velocity (rad/s)

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
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to inference mode

    # Inference loop
    trajectory = []

    for _ in tqdm(range(num_steps)):
        # Perform inference
        with torch.inference_mode():
            output = model(input)

        # Store the output
        trajectory.append(output.cpu().numpy())

        # Update the input for the next step
        input = torch.cat([input[:, :3], output], dim=1)

    # Convert trajectory to a NumPy array
    theta = np.array([state[0][0] for state in trajectory])
    omega = np.array([state[0][1] for state in trajectory])

    return theta, omega
