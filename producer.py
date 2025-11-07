# producer.py
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka_config import BOOTSTRAP_SERVERS, TOPIC

# === Configuración ===
CSV_PATH = "./data/clean_happiness.csv"
FEATURES = [
    "GDP per Capita",
    "Healthy life expectancy",
    "Social support",
    "Freedom",
    "Generosity",
    "Perceptions of corruption",
]
TARGET = "Happiness Score"
ID_COLS = ["Country", "Year"]

# === Productor Kafka ===
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    print(f"Producer: {CSV_PATH} cargado ({df.shape[0]} registros)")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8")
    )

    sent = 0
    for _, row in df.iterrows():
        payload = {
            "meta": {k: row[k] for k in ID_COLS},
            "features": {k: float(row[k]) for k in FEATURES},
            "target": float(row[TARGET]),
        }
        producer.send(TOPIC, value=payload)
        sent += 1
        if sent % 100 == 0:
            print(f"Enviados {sent} mensajes...")
        time.sleep(0.02)

    producer.flush()
    print(f"Producer finalizado. Total enviados: {sent} al tópico '{TOPIC}'.")
