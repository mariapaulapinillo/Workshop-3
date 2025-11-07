

import os
import numpy as np
import pandas as pd
import joblib

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Carga de datos
# =========================
CSV_PATH = "./data/predictions_stream.csv"
MODEL_PATH = "./model/random_forest_model.pkl"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError("No se encontró el archivo predictions_stream.csv.")

df = pd.read_csv(CSV_PATH)

assert {"y_true", "y_pred"}.issubset(df.columns), "El CSV debe contener y_true y y_pred"

if "Country" not in df.columns:
    df["Country"] = "Unknown"
if "Year" not in df.columns:
    df["Year"] = "All"

# Errores y métricas
df["residual"] = df["y_true"] - df["y_pred"]
df["abs_error"] = df["residual"].abs()

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

R2 = r2_score(df["y_true"], df["y_pred"])
MAE = float(df["abs_error"].mean())
RMSE = float(np.sqrt(np.mean(df["residual"] ** 2)))

# Importancia de variables
feature_importance_df = pd.DataFrame(columns=["Feature", "Importance"])
feat_cols_in_csv = [c for c in df.columns if c.startswith("feat_")]
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "feature_importances_"):
            if feat_cols_in_csv:
                pretty = [c.replace("feat_", "").replace("_", " ") for c in feat_cols_in_csv]
                importances = model.feature_importances_
                n = min(len(pretty), len(importances))
                feature_importance_df = pd.DataFrame({
                    "Feature": pretty[:n],
                    "Importance": importances[:n]
                }).sort_values("Importance", ascending=True)
    except Exception:
        pass

years = ["All"] + sorted([str(y) for y in df["Year"].unique() if str(y) != "All"])
countries = ["All"] + sorted(df["Country"].astype(str).unique())

# =========================
# Dash app
# =========================
app = Dash(__name__, title="Model Performance – Happiness")

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui"},
    children=[
        html.H2("Desempeño del Modelo – World Happiness (Streaming)"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"},
            children=[
                html.Div(
                    style={"padding": "14px","border": "1px solid #e5e7eb","borderRadius": "12px",
                           "background": "white"},
                    children=[html.Div("R²", style={"color":"#6b7280","fontSize":"12px"}),
                              html.Div(f"{R2:.3f}", style={"fontSize":"28px","fontWeight":700})],
                ),
                html.Div(
                    style={"padding": "14px","border": "1px solid #e5e7eb","borderRadius": "12px",
                           "background": "white"},
                    children=[html.Div("MAE", style={"color":"#6b7280","fontSize":"12px"}),
                              html.Div(f"{MAE:.3f}", style={"fontSize":"28px","fontWeight":700})],
                ),
                html.Div(
                    style={"padding": "14px","border": "1px solid #e5e7eb","borderRadius": "12px",
                           "background": "white"},
                    children=[html.Div("RMSE", style={"color":"#6b7280","fontSize":"12px"}),
                              html.Div(f"{RMSE:.3f}", style={"fontSize":"28px","fontWeight":700})],
                ),
            ],
        ),

        html.Br(),
        html.Div(
            style={"display": "flex", "gap": "12px"},
            children=[
                html.Div("Año:"),
                dcc.Dropdown(id="year-dd",
                             options=[{"label": y, "value": y} for y in years],
                             value="All", clearable=False, style={"width":"160px"}),
            ],
        ),

        html.Br(),
        # 1) Real vs Predicho
        html.Div(style={"border":"1px solid #e5e7eb","borderRadius":"12px","padding":"8px"},
                 children=[html.H4("1) Real vs. Predicho (con línea y=x)"),
                           dcc.Graph(id="scatter-real-vs-pred")]),

        html.Br(),
        # 2) Mapa de error promedio por país
        html.Div(style={"border":"1px solid #e5e7eb","borderRadius":"12px","padding":"8px"},
                 children=[html.H4("2) Mapa mundial — Error promedio (MAE) por país"),
                           dcc.Graph(id="map-mae")]),

        html.Br(),
        # 3) MAE por año
        html.Div(style={"border":"1px solid #e5e7eb","borderRadius":"12px","padding":"8px"},
                 children=[html.H4("3) MAE por año"),
                           dcc.Graph(id="mae-year")]),

        html.Br(),
        # 4) Importancia de variables
        html.Div(style={"border":"1px solid #e5e7eb","borderRadius":"12px","padding":"8px"},
                 children=[html.H4("4) Importancia de variables (Random Forest)"),
                           dcc.Graph(
                               id="feat-importance",
                               figure=px.bar(feature_importance_df, x="Importance", y="Feature",
                                             orientation="h") if len(feature_importance_df) > 0
                               else go.Figure().add_annotation(x=0.5, y=0.5, text="No disponible",
                                                               showarrow=False))]),
    ],
)

# =========================
# Callbacks
# =========================
@app.callback(
    Output("scatter-real-vs-pred", "figure"),
    Output("map-mae", "figure"),
    Output("mae-year", "figure"),
    Input("year-dd", "value"),
)
def update_figs(year_val):
    d = df.copy()
    if year_val != "All":
        d = d[d["Year"].astype(str) == str(year_val)]

    # 1) Real vs Predicho
    fig1 = px.scatter(d, x="y_true", y="y_pred",
                      color=d["Year"].astype(str),
                      labels={"y_true":"Valor real", "y_pred":"Predicción"})
    mn, mx = float(min(d["y_true"].min(), d["y_pred"].min())), float(max(d["y_true"].max(), d["y_pred"].max()))
    fig1.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                              name="Ideal (y=x)", line=dict(color="red")))
    fig1.update_layout(legend_title_text="Año")

    # 2) Mapa de error promedio (MAE) por país
    geo = d.groupby("Country", as_index=False)["abs_error"].mean()
    fig2 = px.choropleth(
        geo,
        locations="Country",
        locationmode="country names",
        color="abs_error",
        color_continuous_scale="YlOrRd",
        title="Error medio absoluto (MAE) por país",
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))

    # 3) MAE por año
    grp = d.groupby("Year", as_index=False)["abs_error"].mean()
    fig3 = px.bar(grp, x="Year", y="abs_error", text="abs_error",
                  labels={"abs_error":"MAE"})
    fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig3.update_layout(yaxis_title="MAE", xaxis_title="Año")

    return fig1, fig2, fig3



if __name__ == "__main__":
    app.run(debug=True)
