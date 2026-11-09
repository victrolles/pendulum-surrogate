import argparse

import numpy as np
import matplotlib.pyplot as plt

from .evaluation import evaluate
from .inference import infer
from .simulation import _simulate_on_grid, time_grid

def run_simulation(
    l: float = 1.0,             # length of the pendulum (m)
    m: float = 1.0,             # mass of the pendulum (kg)
    b: float = 0.1,             # damping coefficient (kg/s)
    theta0: float = np.pi / 4,  # initial angle (45 degrees)
    omega0: float = 0.0,        # initial angular velocity (rad/s)
    duration: float = 10.0,
    framerate: float = 50.0,
) -> None:

    t = time_grid(duration, framerate)
    theta, omega = _simulate_on_grid(l, m, b, theta0, omega0, t)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, theta, label='Angle (rad)')
    ax.plot(t, omega, label='Angular Velocity (rad/s)')
    plt.title('Pendulum Simulation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Values')
    ax.legend()

    plt.show()

def run_inference(
    model_path: str = "models/model_3992.pth",
    hidden_size: int = 64,
    num_hidden_layers: int = 1,
    l: float = 1.0,
    m: float = 1.0,
    b: float = 0.1,
    theta0: float = np.pi / 4,
    omega0: float = 0.0,
    duration: float = 10.0,
    framerate: float = 50.0,
) -> None:

    t = time_grid(duration, framerate)
    theta, omega = infer(
        model_path=model_path,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_steps=len(t) - 1,
        l=l,
        m=m,
        b=b,
        theta0=theta0,
        omega0=omega0
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t[1:], theta, label='Angle (rad)')
    ax.plot(t[1:], omega, label='Angular Velocity (rad/s)')
    plt.title('Pendulum Simulation using Neural Network Inference')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Values')
    ax.legend()

    plt.show()

def run(args):
    if args.mode == 'simulate':
        run_simulation(
            l=args.l,
            m=args.m,
            b=args.b,
            theta0=args.theta0,
            omega0=args.omega0,
            duration=args.duration,
            framerate=args.framerate,
        )
    elif args.mode == 'infer':
        run_inference(
            model_path=args.model_path,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            l=args.l,
            m=args.m,
            b=args.b,
            theta0=args.theta0,
            omega0=args.omega0,
            duration=args.duration,
            framerate=args.framerate,
        )
    elif args.mode == 'evaluate':
        evaluate(
            model_path=args.model_path,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            l=args.l,
            m=args.m,
            b=args.b,
            theta0=args.theta0,
            omega0=args.omega0,
            duration=args.duration,
            framerate=args.framerate,
        )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pendulum simulation, model inference, or evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mode', choices=['simulate', 'infer', 'evaluate'], default='simulate',
                        help="Mode to run.")
    parser.add_argument('--l', type=float, default=1.0,
                        help="Pendulum length in meters (simulate, infer, and evaluate modes).")
    parser.add_argument('--m', type=float, default=1.0,
                        help="Pendulum mass in kilograms (simulate, infer, and evaluate modes).")
    parser.add_argument('--b', type=float, default=0.1,
                        help="Damping coefficient in kg/s (simulate, infer, and evaluate modes).")
    parser.add_argument('--theta0', type=float, default=np.pi / 4,
                        help="Initial angle in radians (simulate, infer, and evaluate modes).")
    parser.add_argument('--omega0', type=float, default=0.0,
                        help="Initial angular velocity in rad/s (simulate, infer, and evaluate modes).")
    parser.add_argument('--duration', type=float, default=10.0,
                        help="Trajectory duration in seconds.")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--framerate', type=float, default=50.0,
                          help="Trajectory sampling frequency in Hz.")
    sampling.add_argument('--num_points', type=int, help=argparse.SUPPRESS)
    sampling.add_argument('--num_steps', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--model_path', type=str, default="models/model_3992.pth",
                        help="Path to the trained model weights (infer and evaluate modes).")
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Number of neurons in each hidden layer (infer and evaluate modes).")
    parser.add_argument('--num_hidden_layers', type=int, default=1,
                        help="Number of hidden layers in the model (infer and evaluate modes).")

    args = parser.parse_args()
    # Translate the old CLI options to a duration and sampling frequency.
    legacy_steps = args.num_steps if args.num_points is None else args.num_points - 1
    if legacy_steps is not None:
        if legacy_steps < 1 or not np.isfinite(args.duration) or args.duration <= 0:
            parser.error("The duration and number of transitions must be positive.")
        if args.mode == 'infer' and args.num_steps is not None:
            args.duration = legacy_steps / args.framerate
        else:
            args.framerate = legacy_steps / args.duration
    return args

if __name__ == "__main__":
    args = parse_args()
    run(args)
