import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def inference():
    # Simulation parameters
    l = 1.0  # length of the pendulum (m)
    m = 1.0  # mass of the pendulum (kg)
    b = 0.1  # damping coefficient (kg/s)
    theta0 = np.pi / 4  # initial angle (radians)
    omega0 = 0.0  # initial angular velocity (rad/s)

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
    model = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    # Load the trained model weights
    state_dict = torch.load("models/pendulum/model_499.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set the model to inference mode

    # Inference loop
    num_steps = 500
    trajectory = []

    for step in tqdm(range(num_steps)):
        # Perform inference
        with torch.inference_mode():
            output = model(input)

        # Store the output
        trajectory.append(output.cpu().numpy())

        # Update the input for the next step
        input = torch.cat([input[:, :3], output], dim=1)

    # Convert trajectory to a NumPy array
    print(trajectory[0])
    print(trajectory[0][0][0])
    print(trajectory[0][0][1])
    theta = np.array([state[0][0] for state in trajectory])
    omega = np.array([state[0][1] for state in trajectory])

    # plot the results
    t = np.linspace(0, num_steps * 0.02, num_steps)  # Assuming a time step of 0.02s
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, theta, label='Angle (rad)')
    ax.plot(t, omega, label='Angular Velocity (rad/s)')
    plt.title('Pendulum Simulation using Neural Network Inference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Values')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    inference()