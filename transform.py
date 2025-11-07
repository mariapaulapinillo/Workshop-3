
import os
import pandas as pd
from extract import extract_data

COMMON_COLS = [
    "Country", "Year", "Happiness Score",
    "GDP per Capita", "Healthy life expectancy",
    "Social support", "Freedom", "Generosity",
    "Perceptions of corruption"
]

RENAME_MAP = {
    "Country": "Country",
    "Country or region": "Country",
    "Happiness Score": "Happiness Score",
    "Happiness.Score": "Happiness Score",
    "Score": "Happiness Score",
    "Economy (GDP per Capita)": "GDP per Capita",
    "Economy..GDP.per.Capita.": "GDP per Capita",
    "GDP per capita": "GDP per Capita",
    "Health (Life Expectancy)": "Healthy life expectancy",
    "Health..Life.Expectancy.": "Healthy life expectancy",
    "Healthy life expectancy": "Healthy life expectancy",
    "Family": "Social support",
    "Social support": "Social support",
    "Freedom": "Freedom",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Trust (Government Corruption)": "Perceptions of corruption",
    "Trust..Government.Corruption.": "Perceptions of corruption",
    "Perceptions of corruption": "Perceptions of corruption",
}

def to_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

def transform(path="./data/", years=None) -> pd.DataFrame:
    

    if years is None:
        years = [2015, 2016, 2017, 2018, 2019]

    files = {y: os.path.join(path, f"{y}.csv") for y in years}
    raw = extract_data(files)

    # Estandarizar columnas y asegurar Year
    for y, d in raw.items():
        if "Year" not in d.columns:
            d["Year"] = y
        raw[y] = to_common_columns(d)

    # Concatenar datasets
    frames = []
    for y in years:
        if y in raw:
            cols_ok = [c for c in COMMON_COLS if c in raw[y].columns]
            frames.append(raw[y][cols_ok])
    df_all = pd.concat(frames, ignore_index=True)
    
    
   
    
    if "Perceptions of corruption" in df_all.columns:
        df_all["Perceptions of corruption"] = df_all.groupby("Year")["Perceptions of corruption"]\
            .transform(lambda s: s.fillna(s.median()))
        

   
    
    return df_all


if __name__ == "__main__":
    df_all = transform("./data/")
