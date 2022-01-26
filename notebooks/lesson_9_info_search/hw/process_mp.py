
import pandas as pd
import langid
import time
from concurrent import futures
import numpy as np
import multiprocessing as mp


def is_en_bool(text):
    return langid.classify(text)[0] == 'en'

def process_series(series: pd.Series):
    return series.map(is_en_bool)

def main_n_process(parts):
    processes = [
        mp.Process(target=process_series, args=(part,))
        for part in parts
    ]
    
    for process in processes:
        process.start()

    print("START TIME")
    start = time.perf_counter()
    
    for process in processes:
        process.join()

    print(f"END TIME: {time.perf_counter() - start}")
    return process

if __name__ == '__main__':
    slice_data = pd.read_csv("github_issues_slice.csv")[:1000]
    n = 4
    parts = np.array_split(slice_data.body, n)
    result = main_n_process(parts)
    print(result)
