
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATA_PATH = "./data/clean_happiness.csv"
MODEL_DIR = "./model"
os.makedirs(MODEL_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape}")

features = [
    "GDP per Capita",
    "Healthy life expectancy",
    "Social support",
    "Freedom",
    "Generosity",
    "Perceptions of corruption",
]
target = "Happiness Score"

X = df[features].to_numpy()
y = df[target].to_numpy().reshape(-1, 1)
n, p = X.shape

# Dividir datos 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Conjunto de entrenamiento: {X_train.shape}, prueba: {X_test.shape}")

# Modelo 1: OLS Regresión Lineal 
X_train_design = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_design = np.c_[np.ones(X_test.shape[0]), X_test]

# β̂ = (XᵀX)⁻¹ XᵀY
beta_hat = np.linalg.inv(X_train_design.T @ X_train_design) @ X_train_design.T @ y_train

# Predicciones
y_pred_train = X_train_design @ beta_hat
y_pred_test = X_test_design @ beta_hat

# Métricas OLS
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

r2_ols, mae_ols, rmse_ols = metrics(y_test, y_pred_test)

print(" Modelo 1: Regresión Lineal (OLS)")
print("Coeficientes β̂:")
for name, coef in zip(["Intercept"] + features, beta_hat.flatten()):
    print(f"  {name:30s}: {coef:8.4f}")
print(f"R² = {r2_ols:.3f}, MAE = {mae_ols:.3f}, RMSE = {rmse_ols:.3f}")

# Guardar modelo OLS
joblib.dump({"beta": beta_hat, "features": ["Intercept"] + features}, os.path.join(MODEL_DIR, "ols_model.pkl"))

#  Modelo 2: Random Forest 
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train.ravel())
y_pred_rf = rf.predict(X_test)

r2_rf, mae_rf, rmse_rf = metrics(y_test, y_pred_rf)
print("Modelo 2: Random Forest Regressor")
print(f"R² = {r2_rf:.3f}, MAE = {mae_rf:.3f}, RMSE = {rmse_rf:.3f}")


joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

#  Comparación final
print(" Comparación de desempeño:")
print(pd.DataFrame({
    "Modelo": ["OLS (Lineal)", "Random Forest"],
    "R²": [r2_ols, r2_rf],
    "MAE": [mae_ols, mae_rf],
    "RMSE": [rmse_ols, rmse_rf]
}))
