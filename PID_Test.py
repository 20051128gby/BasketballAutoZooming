import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
g = 9.81  # Gravitational acceleration (m/s^2)
l = 1.0   # Length of the pendulum (m)
theta_0 = 0.2  # Initial angle (radians)
omega_0 = 0.0  # Initial angular velocity (rad/s)
T = 10  # Total time (seconds)
h = 0.05  # Step size

# Time array
t = np.arange(0, T + h, h)

# Initialize arrays for theta (angle) and omega (angular velocity)
theta_euler = np.zeros(len(t))
omega_euler = np.zeros(len(t))
theta_improved_euler = np.zeros(len(t))
omega_improved_euler = np.zeros(len(t))
theta_rk = np.zeros(len(t))
omega_rk = np.zeros(len(t))

# Initial conditions
theta_euler[0] = theta_0
omega_euler[0] = omega_0
theta_improved_euler[0] = theta_0
omega_improved_euler[0] = omega_0
theta_rk[0] = theta_0
omega_rk[0] = omega_0

# Euler's Method
for n in range(len(t) - 1):
    omega_euler[n + 1] = omega_euler[n] - (g / l) * np.sin(theta_euler[n]) * h
    theta_euler[n + 1] = theta_euler[n] + omega_euler[n] * h

# Improved Euler's Method
for n in range(len(t) - 1):
    omega_half = omega_improved_euler[n] - (g / l) * np.sin(theta_improved_euler[n]) * h / 2
    theta_half = theta_improved_euler[n] + omega_improved_euler[n] * h / 2
    omega_improved_euler[n + 1] = omega_improved_euler[n] - (g / l) * np.sin(theta_half) * h
    theta_improved_euler[n + 1] = theta_improved_euler[n] + (omega_improved_euler[n] + omega_improved_euler[n + 1]) * h / 2

# Runge-Kutta Method
for n in range(len(t) - 1):
    k1_omega = - (g / l) * np.sin(theta_rk[n]) * h
    k1_theta = omega_rk[n] * h
    k2_omega = - (g / l) * np.sin(theta_rk[n] + k1_theta / 2) * h
    k2_theta = (omega_rk[n] + k1_omega / 2) * h
    k3_omega = - (g / l) * np.sin(theta_rk[n] + k2_theta / 2) * h
    k3_theta = (omega_rk[n] + k2_omega / 2) * h
    k4_omega = - (g / l) * np.sin(theta_rk[n] + k3_theta) * h
    k4_theta = (omega_rk[n] + k3_omega) * h
    
    omega_rk[n + 1] = omega_rk[n] + (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6
    theta_rk[n + 1] = theta_rk[n] + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6

# Analytical solution (small angle approximation)
theta_analytical = theta_0 * np.cos(np.sqrt(g / l) * t)

# Plot results with adjusted line thickness (linewidth)
plt.figure(figsize=(10, 6))
plt.plot(t, theta_analytical, label="Analytical Solution", color='black', linestyle='--', linewidth=2)
plt.plot(t, theta_euler, label="Euler's Method", color='red', linewidth=2)
plt.plot(t, theta_improved_euler, label="Improved Euler Method", color='green', linewidth=2)
plt.plot(t, theta_rk, label="Runge-Kutta Method", color='blue', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('Comparison of Pendulum Solutions')
plt.grid(True)
plt.show()

# Error calculation
error_euler = np.abs(theta_analytical - theta_euler)
error_improved_euler = np.abs(theta_analytical - theta_improved_euler)
error_rk = np.abs(theta_analytical - theta_rk)

# Error plot with adjusted line thickness (linewidth)
plt.figure(figsize=(10, 6))
plt.plot(t, error_euler, label="Euler's Method Error", color='red', linewidth=2)
plt.plot(t, error_improved_euler, label="Improved Euler Method Error", color='green', linewidth=2)
plt.plot(t, error_rk, label="Runge-Kutta Method Error", color='blue', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error (rad)')
plt.legend()
plt.title('Error Comparison of Numerical Methods')
plt.grid(True)
plt.show()

# Create two separate tables: one for numerical solutions, and one for errors

# Numerical solution table
numerical_data = {
    'Time (s)': t,
    'Euler\'s Method': theta_euler,
    'Improved Euler Method': theta_improved_euler,
    'Runge-Kutta Method': theta_rk
}
numerical_df = pd.DataFrame(numerical_data)
print("Numerical Solution Table:")
print(numerical_df.head())  # Show first few rows of numerical solution table

# Error table
error_data = {
    'Time (s)': t,
    'Euler\'s Method Error': error_euler,
    'Improved Euler Method Error': error_improved_euler,
    'Runge-Kutta Method Error': error_rk
}
error_df = pd.DataFrame(error_data)
print("\nError Table:")
print(error_df.head())  # Show first few rows of error table
