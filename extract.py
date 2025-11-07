import pandas as pd
import numpy as np 

DATA_PATH = "./data/"
YEARS = [2015, 2016, 2017, 2018, 2019]
FILES = {y: f"{DATA_PATH}{y}.csv" for y in YEARS}

def extract_data(files):
    datasets = {}
    for y, path in files.items():
        df = pd.read_csv(path)
        df["Year"] = y
        datasets[y] = df
        print(f"{y}: {df.shape}")
    return datasets

raw = extract_data(FILES)