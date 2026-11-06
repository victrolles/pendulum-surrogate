import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def main():
    # 1. Physical parameters
    g = 9.81  # acceleration due to gravity (m/s^2)
    l = 1.0   # length of the pendulum (m)
    m = 1.0   # mass of the pendulum (kg)

    # 2. Initial conditions
    theta0 = np.radians(45)  # initial angle (radians)
    omega0 = 0.0              # initial angular velocity (rad/s)

    # 3. Physical equation of the pendulum
    def pendulum(t, y):
        theta, omega = y
        dtheta = omega
        domega = -(g / l) * np.sin(theta)
        return [dtheta, domega]

    # 4. Simulation time
    # Time interval for the simulation (start, end, step)
    t = np.linspace(0, 10, 500)  # simulate for 10 seconds

    # 5. Solve the differential equations
    solution = solve_ivp(
        pendulum, 
        [t[0], t[-1]],
        [theta0, omega0],
        t_eval=t
    )

    # 6. Plotting the results
    theta = solution.y[0]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    ax.plot(t, theta)
    ax.set_title('Pendulum Motion')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (radians)')

    plt.show()

if __name__ == "__main__":
    main()