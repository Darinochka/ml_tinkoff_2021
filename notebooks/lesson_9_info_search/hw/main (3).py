import pandas as pd
# import modin.pandas as pd
import langid
import time
from concurrent import futures
import numpy as np



def is_en(text):
    return text if langid.classify(text)[0] == 'en' else pd.NA

def en_condition(row):
    return langid.classify(row)[0] == 'en'

def main_naive(df: pd.DataFrame):
    start = time.perf_counter()
    print("STARTING TIMER NAIVE")


    to_delete = df[df.body.map(en_condition)]
    df = df.drop(to_delete.index)


    print("TIMER ENDED", time.perf_counter() - start)

def main_apply(df: pd.DataFrame):
    start = time.perf_counter()
    print("STARTING TIMER APPLY")
    res = df.body.apply(is_en)
    res.dropna()
    print("TIMER ENDED", time.perf_counter() - start)

def main_n_process(df: pd.DataFrame):
    start = time.perf_counter()
    print("STARTING TIMER NAIVE")

    n = 4
    parts = np.array_split(df.body, n)

    with futures.ProcessPoolExecutor() as executor:
        res = executor.map(process_series, parts)
    

    print("TIMER ENDED", time.perf_counter() - start)


import multiprocessing as mp

def process_series(series: pd.Series):
    return series.map(en_condition)

def main_n_process_manual(df: pd.DataFrame):
    n = 4
    parts = np.array_split(df.body, n)

    processes = [
        mp.Process(target=process_series, args=(part,))
        for part in parts
    ]
    
    for process in processes:
        process.start()
    
    start = time.perf_counter()
    print("STARTING TIMER MANUAL")
    
    for process in processes:
        process.join()

    print("TIMER ENDED", time.perf_counter() - start)

def main_process_pool_executor(df: pd.DataFrame):
    start = time.perf_counter()
    print("STARTING TIMER PROCESS")

    with futures.ProcessPoolExecutor() as executor:
        res = pd.Series(executor.map(is_en, df.body))

    print("PROCESS ENDED AFTER", time.perf_counter() - start)
    print(len(res.dropna()))
    print("DROPPING ENDED AFTER ", time.perf_counter() - start)

def main_thread_pool_executor(df: pd.DataFrame):
    start = time.perf_counter()
    print("STARTING TIMER THREADS")

    with futures.ThreadPoolExecutor() as executor:
        res = pd.Series(executor.map(is_en, df.body))

    print("PROCESS ENDED AFTER", time.perf_counter() - start)
    print(len(res.dropna()))
    print("DROPPING ENDED AFTER ", time.perf_counter() - start)

if __name__ == '__main__':
    df = pd.read_csv("notebooks\lesson_9_info_search\hw\github_issues_slice.csv")[:100000]

    # main_naive(df)
    main_n_process(df)
    # main_n_process_manual(df)
    # main_apply(df)
    # main_process_pool_executor(df)
    # main_thread_pool_executor(df)
