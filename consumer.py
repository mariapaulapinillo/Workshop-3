# consumer.py
import os
import json
import sqlite3
import joblib
import pandas as pd
from kafka import KafkaConsumer
from kafka_config import BOOTSTRAP_SERVERS, TOPIC
import warnings
warnings.filterwarnings("ignore", message="X has feature names")


# === Configuración ===
MODEL_PATH = "./model/random_forest_model.pkl"
CSV_OUT = "./data/predictions_stream.csv"
DB_PATH = "./data/predictions.db"
TABLE = "predictions"

FEATURES = [
    "GDP per Capita",
    "Healthy life expectancy",
    "Social support",
    "Freedom",
    "Generosity",
    "Perceptions of corruption",
]
ID_COLS = ["Country", "Year"]

def append_csv(row_dict):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(CSV_OUT):
        df.to_csv(CSV_OUT, index=False)
    else:
        df.to_csv(CSV_OUT, mode="a", header=False, index=False)

def insert_sqlite(row_dict):
    conn = sqlite3.connect(DB_PATH)
    pd.DataFrame([row_dict]).to_sql(TABLE, conn, if_exists="append", index=False)
    conn.close()

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./model", exist_ok=True)

    print("Consumer: cargando modelo...")
    model = joblib.load(MODEL_PATH)
    print(f" Modelo cargado desde: {MODEL_PATH}")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    print(f"Escuchando el tópico '{TOPIC}'...\n")

    count = 0
    for msg in consumer:
        try:
            payload = msg.value
            meta = payload.get("meta", {})
            feats = payload["features"]
            y_true = payload["target"]

            X = pd.DataFrame([[feats[k] for k in FEATURES]], columns=FEATURES)
            y_pred = float(model.predict(X)[0])

            row_out = {
                **meta,
                "y_true": y_true,
                "y_pred": y_pred,
                **{f"feat_{k.replace(' ', '_')}": feats[k] for k in FEATURES},
            }

            append_csv(row_out)
            insert_sqlite(row_out)
            count += 1

            if count % 100 == 0:
                print(f"Procesados {count} mensajes...")
        except Exception as e:
            print("⚠️ Error procesando mensaje:", e)
