import numpy as np
from scipy.integrate import solve_ivp

G = 9.81  # acceleration due to gravity (m/s^2)

def simulate(
        l: float = 1.0,       # length of the pendulum (m)
        m: float = 1.0,       # mass of the pendulum (kg)
        b: float = 0.2,       # damping coefficient (kg/s)
        theta0: float = 0.0,  # initial angle (radians)
        omega0: float = 0.0,  # initial angular velocity (rad/s)
        t: np.ndarray = np.linspace(0, 10, 1000)  # time array (s)
    ) -> list[np.ndarray]:

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