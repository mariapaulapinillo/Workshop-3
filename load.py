
import os
import pandas as pd
from transform import transform   

if __name__ == "__main__":
    print(" Ejecutando Load (solo CSV)...\n")


    df = transform(path="./data/")

    
    output_path = os.path.join("./data", "clean_happiness.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n CSV guardado exitosamente en: {output_path}")
    print(f"Filas: {len(df)}, Columnas: {df.shape[1]}")
    print("\nProceso Load finalizado.")
