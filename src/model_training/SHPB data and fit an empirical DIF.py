import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0 and "strength" not in prompt.lower():
                print("Value must be positive.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

print("Enter the following parameters for the SHPB setup:")
E_bar = get_float_input("Young's modulus of bar (Pa): ")
A_bar = get_float_input("Cross-sectional area of bar (m^2): ")
A_specimen = get_float_input("Cross-sectional area of specimen (m^2): ")
L_specimen = get_float_input("Length of specimen (m): ")
c0 = get_float_input("Wave speed in bar (m/s): ")
static_strength = get_float_input("Static strength of material (Pa): ")
L_bar = get_float_input("Length of bar (m): ")
k = get_float_input("Calibration factor (strain/V): ")

excel_path = input("Enter the path to your Excel file: ")
try:
    data = pd.read_excel(excel_path)
    
    required_columns = ["Time (s) - PXI1Slot4/ai0", "Voltage (V) - PXI1Slot4/ai0",
                        "Time (s) - PXI1Slot4/ai1", "Voltage (V) - PXI1Slot4/ai1"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Excel file must contain the required columns: " + ", ".join(required_columns))
except FileNotFoundError:
    print(f"Error: File '{excel_path}' not found.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()

voltage_inc_bar = data["Voltage (V) - PXI1Slot4/ai0"].values
voltage_trans_bar = data["Voltage (V) - PXI1Slot4/ai1"].values
time = data["Time (s) - PXI1Slot4/ai0"].values

strain_inc_bar = voltage_inc_bar * k  
strain_trans_bar = voltage_trans_bar * k  

t_separation = L_bar / c0 
mask_incident = time <= t_separation
mask_reflected = time > t_separation

eps_i = strain_inc_bar * mask_incident  
eps_r = strain_inc_bar * mask_reflected  
eps_t = strain_trans_bar  

dt = time[1] - time[0] 
strain_rate = -2 * c0 * eps_r / L_specimen
strain = np.cumsum(strain_rate) * dt
stress = E_bar * A_bar * eps_t / A_specimen
avg_strain_rate = np.mean(strain_rate[strain_rate > 0]) if np.any(strain_rate > 0) else 0
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
avg_strain_rate_pred = np.mean(strain_rate_pred[strain_rate_pred > 0]) if np.any(strain_rate_pred > 0) else 0
dynamic_strength_pred = np.max(stress_pred)
DIF_pred = dynamic_strength_pred / static_strength

print(f"\nExperimental - Avg strain rate: {avg_strain_rate:.2f} s^-1, Dynamic strength: {dynamic_strength/1e6:.2f} MPa, DIF: {DIF:.2f}")
print(f"Predicted - Avg strain rate: {avg_strain_rate_pred:.2f} s^-1, Dynamic strength: {dynamic_strength_pred/1e6:.2f} MPa, DIF: {DIF_pred:.2f}")

results_df = pd.DataFrame({
    "Time (s)": time,
    "Strain (Experimental)": strain,
    "Stress (MPa, Experimental)": stress / 1e6,
    "Strain Rate (s^-1, Experimental)": strain_rate,
    "Strain (Predicted)": strain_pred,
    "Stress (MPa, Predicted)": stress_pred / 1e6,
    "Strain Rate (s^-1, Predicted)": strain_rate_pred
})
results_df.to_csv("C:/Users/Busiso/Desktop/shpb_results.csv", index=False)
print("\nResults saved to 'C:/Users/Busiso/Desktop/shpb_results.csv'")

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
plt.scatter([avg_strain_rate_pred], [DIF_pred], label="Predicted DIF", color="red")
plt.xscale("log")
plt.xlabel("Strain Rate (s^-1)")
plt.ylabel("DIF")
plt.title("Dynamic Increase Factor")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

joblib.dump(model, "C:/Users/Busiso/Desktop/shpb_digital_twin_model.pkl")
joblib.dump(scaler_X, "C:/Users/Busiso/Desktop/scaler_X.pkl")
joblib.dump(scaler_y, "C:/Users/Busiso/Desktop/scaler_y.pkl")