# evaluate_model.py  (sin scikit-learn)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "./data/predictions_stream.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"No existe {CSV_PATH}. Ejecuta antes consumer.py.")

df = pd.read_csv(CSV_PATH)
print(f" Archivo cargado: {CSV_PATH} ({len(df)} filas)")

y_true = df["y_true"].to_numpy(dtype=float)
y_pred = df["y_pred"].to_numpy(dtype=float)

# --- Métricas manuales ---
def r2_score_manual(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

def mae_manual(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse_manual(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

r2   = r2_score_manual(y_true, y_pred)
mae  = mae_manual(y_true, y_pred)
rmse = rmse_manual(y_true, y_pred)

print("\n Métricas de desempeño (Random Forest en streaming)")
print(f"R²   : {r2:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")

# --- Gráfica: Predicho vs Real ---
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()], 'r-', label="Ideal (y=x)")
plt.xlabel("Valor real (Happiness Score)")
plt.ylabel("Predicción")
plt.title("Predicción vs. Realidad — Streaming")
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfica: Distribución de errores ---
errors = y_true - y_pred
plt.figure(figsize=(7,5))
sns.histplot(errors, kde=True, bins=25)
plt.title("Distribución del error (y_real - y_pred)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()
