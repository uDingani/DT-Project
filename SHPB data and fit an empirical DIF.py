import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Simulated raw data (replace with your actual file)
data = pd.DataFrame({
    "Time (s) - PXI1Slot4/ai0": [0.0, 5e-07],
    "Voltage (V) - PXI1Slot4/ai0": [4.30811, 4.31749],
    "Time (s) - PXI1Slot4/ai1": [0.0, 5e-07],
    "Voltage (V) - PXI1Slot4/ai1": [3.62989, 3.63161]
})

# Parameters (hardcoded for now, replace with your values)
E_bar = 200e9  # Pa
A_bar = 1.26e-4  # m^2
A_specimen = 7.85e-5  # m^2
L_specimen = 0.005  # m
c0 = 5000  # m/s
static_strength = 50e6  # Pa
L_bar = 1.0  # m (assumed incident bar length)

# Placeholder voltage-to-strain conversion (adjust k based on your calibration)
k = 1e-4  # strain/V (example)
voltage_inc_bar = data["Voltage (V) - PXI1Slot4/ai0"].values
voltage_trans_bar = data["Voltage (V) - PXI1Slot4/ai1"].values
time = data["Time (s) - PXI1Slot4/ai0"].values

# Convert voltages to strains
strain_inc_bar = voltage_inc_bar * k  # Incident bar strain (incident + reflected)
strain_trans_bar = voltage_trans_bar * k  # Transmitted bar strain

# Separate incident and reflected pulses (simplified assumption)
# Assume incident pulse ends and reflected begins after t = L_bar/c0 (one-way trip)
t_separation = L_bar / c0  # e.g., 2e-4 s
mask_incident = time <= t_separation
mask_reflected = time > t_separation

eps_i = strain_inc_bar * mask_incident  # Incident strain (zero after separation)
eps_r = strain_inc_bar * mask_reflected  # Reflected strain (zero before separation)
eps_t = strain_trans_bar  # Transmitted strain


dt = time[1] - time[0]
strain_rate = -2 * c0 * eps_r / L_specimen
strain = np.cumsum(strain_rate) * dt
stress = E_bar * A_bar * eps_t / A_specimen
avg_strain_rate = np.mean(strain_rate[strain_rate > 0])
dynamic_strength = np.max(stress)
DIF = dynamic_strength / static_strength

inputs = np.array([E_bar, A_bar, A_specimen, L_specimen, c0, static_strength]).reshape(1, -1)
X = np.column_stack((inputs.repeat(len(time), axis=0), eps_i, eps_r, eps_t))
y = np.column_stack((stress, strain, strain_rate))

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_scaled, y_scaled)

y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
stress_pred, strain_pred, strain_rate_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
dynamic_strength_pred = np.max(stress_pred)
DIF_pred = dynamic_strength_pred / static_strength


print(f"Experimental - Avg strain rate: {avg_strain_rate:.2f} s^-1, Dynamic strength: {dynamic_strength/1e6:.2f} MPa, DIF: {DIF:.2f}")
print(f"Predicted - Avg strain rate: {np.mean(strain_rate_pred[strain_rate_pred > 0]):.2f} s^-1, Dynamic strength: {dynamic_strength_pred/1e6:.2f} MPa, DIF: {DIF_pred:.2f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(strain, stress / 1e6, label="Experimental")
plt.plot(strain_pred, stress_pred / 1e6, '--', label="Predicted")
plt.xlabel("Strain")
plt.ylabel("Stress (MPa)")
plt.title("Stress-Strain Curve")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter([avg_strain_rate], [DIF], label="Experimental DIF", color="blue")
plt.scatter([np.mean(strain_rate_pred[strain_rate_pred > 0])], [DIF_pred], label="Predicted DIF", color="red")
plt.xscale("log")
plt.xlabel("Strain Rate (s^-1)")
plt.ylabel("DIF")
plt.title("Dynamic Increase Factor")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


joblib.dump(model, "shpb_digital_twin_model.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")