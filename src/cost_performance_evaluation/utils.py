import time

def measure_time(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    return result, end_time - start_time
