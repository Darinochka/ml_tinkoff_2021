
import pandas as pd
import langid
import time
from concurrent import futures
import numpy as np

def is_en(text):
    return langid.classify(text)[0] == 'en'

def process_series(series: pd.Series):
    return series.map(is_en)

def main(df):
    n = 4
    parts = np.array_split(df.body, n)

    print("START TIME")
    start = time.perf_counter()

    with futures.ProcessPoolExecutor() as executor:
        res = executor.map(process_series, parts)

    print(f"END TIME: {time.perf_counter() - start}")
    print(res)


if __name__ == '__main__':
    df = pd.read_csv("github_issues_slice.csv")[:1000]

    main(df)
