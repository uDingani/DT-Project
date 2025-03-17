import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Input function
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


print("Enter SHPB Parameters:")
E_bar = get_positive_float("Young's modulus of bars (Pa) [e.g., 200e9 for steel]: ")
A_bar = get_positive_float("Cross-sectional area of bars (m^2) [e.g., 1.26e-4 for 12mm dia]: ")
A_specimen = get_positive_float("Specimen cross-sectional area (m^2) [e.g., 7.85e-5 for 10mm dia]: ")
L_specimen = get_positive_float("Specimen length (m) [e.g., 0.005]: ")
c0 = get_positive_float("Wave speed in bars (m/s) [e.g., 5000]: ")
static_strength = get_positive_float("Static strength (Pa) [e.g., 50e6]: ")


data = pd.read_csv("shpb_data.csv")
time = data["time"].values
eps_i = data["eps_incident"].values
eps_r = data["eps_reflected"].values
eps_t = data["eps_transmitted"].values
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


print(f"Experimental - Average strain rate: {avg_strain_rate:.2f} s^-1, Dynamic strength: {dynamic_strength/1e6:.2f} MPa, DIF: {DIF:.2f}")
print(f"Predicted - Average strain rate: {np.mean(strain_rate_pred[strain_rate_pred > 0]):.2f} s^-1, Dynamic strength: {dynamic_strength_pred/1e6:.2f} MPa, DIF: {DIF_pred:.2f}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(strain, stress / 1e6, label="Experimental")
plt.plot(strain_pred, stress_pred / 1e6, '--', label="Predicted")
plt.xlabel("Strain")
plt.ylabel("Stress (MPa)")
plt.title("Stress-Strain Curve")
plt.grid(True)
plt.legend()


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