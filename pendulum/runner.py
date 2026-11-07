import argparse

import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .simulation import simulate
    from .inference import infer
    from .trainer import train
    from .generation import generate_dataset
else:
    from simulation import simulate
    from inference import infer
    from trainer import train
    from generation import generate_dataset

def run_simulation(
    l: float = 1.0,             # length of the pendulum (m)
    m: float = 1.0,             # mass of the pendulum (kg)
    b: float = 0.1,             # damping coefficient (kg/s)
    theta0: float = np.pi / 4,  # initial angle (45 degrees)
    omega0: float = 0.0,        # initial angular velocity (rad/s)
    t: np.ndarray = np.linspace(0, 20, 1000)
) -> None:

    theta, omega = simulate(
        l=l,
        m=m,
        b=b,
        theta0=theta0,
        omega0=omega0,
        t=t
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, theta, label='Angle (rad)')
    ax.plot(t, omega, label='Angular Velocity (rad/s)')
    plt.title('Pendulum Simulation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Values')
    ax.legend()

    plt.show()

def run_inference(
    model_path: str = "pendulum/models/model_3992.pth",
    hidden_size: int = 64,
    num_hidden_layers: int = 1,
    num_steps: int = 1000,
    l: float = 1.0,
    m: float = 1.0,
    b: float = 0.1,
    theta0: float = np.pi / 4,
    omega0: float = 0.0
) -> None:

    theta, omega = infer(
        model_path=model_path,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_steps=num_steps,
        l=l,
        m=m,
        b=b,
        theta0=theta0,
        omega0=omega0
    )

    t = np.linspace(0, num_steps * 0.02, num_steps)  # Assuming a time step of 0.02s
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, theta, label='Angle (rad)')
    ax.plot(t, omega, label='Angular Velocity (rad/s)')
    plt.title('Pendulum Simulation using Neural Network Inference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Values')
    ax.legend()

    plt.show()

def run_training(
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        learning_rate: float = 0.001,
        train_batch_size: int = 64,
        eval_batch_size: int = 64,
        num_epochs: int = 50,
        model_dir: str = "pendulum/models",
        dataset_dir: str = "datasets/pendulum",
        t: np.ndarray = np.linspace(0, 10, 500)
) -> None:
    # Hyperparameters
    list_l = [1.0, 2.0]        # length of the pendulum (m)
    list_m = [1.0, 2.0]        # mass of the pendulum (kg)
    list_b = [0.1, 0.2]        # damping coefficient (kg/s)
    list_theta0 = [np.pi / 4]  # initial angles (radians)
    list_omega0 = [0.0]        # initial angular velocities (rad/s)

    # Generate the dataset
    dataset_path = generate_dataset(
        dataset_dir=dataset_dir,
        list_l=list_l,
        list_m=list_m,
        list_b=list_b,
        list_theta0=list_theta0,
        list_omega0=list_omega0,
        t=t
    )

    # Train the model
    train(
        dataset_path=dataset_path,
        model_dir=model_dir,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs
    )

def run(args):
    if args.mode == 'simulate':
        run_simulation(
            l=args.l,
            m=args.m,
            b=args.b,
            theta0=args.theta0,
            omega0=args.omega0,
            t=np.linspace(0, args.duration, args.num_points)
        )
    elif args.mode == 'infer':
        run_inference(
            model_path=args.model_path,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_steps=args.num_steps,
            l=args.l,
            m=args.m,
            b=args.b,
            theta0=args.theta0,
            omega0=args.omega0
        )
    elif args.mode == 'train':
        run_training(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            t=np.linspace(0, args.duration, args.num_points)
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pendulum simulation, model inference, or training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mode', type=str, choices=['simulate', 'infer', 'train'], default='simulate',
                        help="Mode to run: 'simulate' for simulation, 'infer' for inference, 'train' for training.")
    parser.add_argument('--l', type=float, default=1.0,
                        help="Pendulum length in meters (simulate and infer modes).")
    parser.add_argument('--m', type=float, default=1.0,
                        help="Pendulum mass in kilograms (simulate and infer modes).")
    parser.add_argument('--b', type=float, default=0.1,
                        help="Damping coefficient in kg/s (simulate and infer modes).")
    parser.add_argument('--theta0', type=float, default=np.pi / 4,
                        help="Initial angle in radians (simulate and infer modes).")
    parser.add_argument('--omega0', type=float, default=0.0,
                        help="Initial angular velocity in rad/s (simulate and infer modes).")
    parser.add_argument('--duration', type=float, default=20.0,
                        help="Simulation duration in seconds (simulate and train modes).")
    parser.add_argument('--num_points', type=int, default=1000,
                        help="Number of time points per simulation (simulate and train modes).")
    parser.add_argument('--model_path', type=str, default="pendulum/models/model_3992.pth",
                        help="Path to the trained model weights to load in infer mode.")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Number of neurons in each hidden layer (infer and train modes).")
    parser.add_argument('--num_hidden_layers', type=int, default=1,
                        help="Number of hidden layers in the model (infer and train modes).")
    parser.add_argument('--num_steps', type=int, default=1000,
                        help="Number of prediction steps (infer mode).")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the Adam optimizer (train mode).")
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help="Number of samples per training batch (train mode).")
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help="Number of samples per validation batch (train mode).")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of training epochs (train mode).")
    parser.add_argument('--model_dir', type=str, default="pendulum/models",
                        help="Directory where the trained model is saved (train mode).")
    parser.add_argument('--dataset_dir', type=str, default="datasets/pendulum",
                        help="Directory where the generated dataset is saved (train mode).")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
