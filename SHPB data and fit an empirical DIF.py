# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:09:44 2025

@author: Busiso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Input validation function
def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Value must be positive. Try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number.")

# Get SHPB parameters
print("Enter SHPB Parameters:")
E_bar = get_positive_float("Young's modulus of bars (Pa) [e.g., 200e9 for steel]: ")
A_bar = get_positive_float("Cross-sectional area of bars (m^2) [e.g., 1.26e-4 for 12mm dia]: ")
A_specimen = get_positive_float("Specimen cross-sectional area (m^2) [e.g., 7.85e-5 for 10mm dia]: ")
L_specimen = get_positive_float("Specimen length (m) [e.g., 0.005]: ")
c0 = get_positive_float("Wave speed in bars (m/s) [e.g., 5000]: ")
static_strength = get_positive_float("Static strength (Pa) [e.g., 50e6]: ")

# Simulated waveforms (replace with real data in practice)
time = np.linspace(0, 200e-6, 1000)
eps_i = 1e-3 * np.sin(2 * np.pi * 5000 * time) * np.exp(-time / 50e-6)
eps_r = -0.5e-3 * np.sin(2 * np.pi * 5000 * time) * np.exp(-time / 50e-6)
eps_t = 0.3e-3 * np.sin(2 * np.pi * 5000 * time) * np.exp(-time / 50e-6)

# Calculations
dt = time[1] - time[0]
strain_rate = -2 * c0 * eps_r / L_specimen
strain = np.cumsum(strain_rate) * dt
stress = E_bar * A_bar * eps_t / A_specimen

avg_strain_rate = np.mean(strain_rate[strain_rate > 0])
dynamic_strength = np.max(stress)
DIF = dynamic_strength / static_strength

# DIF model
def dif_model(strain_rate, A, B, ref_strain_rate=1.0):
    """Model for Dynamic Increase Factor (DIF)."""
    return A + B * np.log10(strain_rate / ref_strain_rate)

# Sample data for fitting (replace with experimental data)
strain_rates = np.array([100, 500, 1000, 2000, 5000])
DIFs = np.array([1.1, 1.3, 1.5, 1.7, 2.0])

# Add calculated point
strain_rates = np.append(strain_rates, avg_strain_rate)
DIFs = np.append(DIFs, DIF)

# Fit DIF model
try:
    popt, pcov = curve_fit(dif_model, strain_rates, DIFs, p0=[1.0, 0.1])
    A_fit, B_fit = popt
except RuntimeError:
    print("Error: Could not fit DIF model to data.")
    A_fit, B_fit = 1.0, 0.0

# Output results
print(f"Fitted parameters: A = {A_fit:.2f}, B = {B_fit:.2f}")
print(f"Average strain rate: {avg_strain_rate:.2f} s^-1")
print(f"Dynamic strength: {dynamic_strength/1e6:.2f} MPa")
print(f"DIF: {DIF:.2f}")

# Plotting
plt.figure(figsize=(12, 5))

# Stress-Strain Plot
plt.subplot(1, 2, 1)
plt.plot(strain, stress / 1e6, label="Stress-Strain")
plt.xlabel("Strain")
plt.ylabel("Stress (MPa)")
plt.title("SHPB Stress-Strain Curve")
plt.grid(True)
plt.legend()

# DIF Plot
plt.subplot(1, 2, 2)
plt.scatter(strain_rates[:-1], DIFs[:-1], label="Sample Data", color="blue")
plt.scatter([avg_strain_rate], [DIF], color="green", label="Calculated DIF", zorder=5)
fit_strain_rates = np.logspace(2, 4, 100)
plt.plot(fit_strain_rates, dif_model(fit_strain_rates, A_fit, B_fit),
         label=f"Fit: DIF = {A_fit:.2f} + {B_fit:.2f}·log(ε̇)", color="red")
plt.xscale("log")
plt.xlabel("Strain Rate (s^-1)")
plt.ylabel("DIF")
plt.title("Dynamic Increase Factor")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()