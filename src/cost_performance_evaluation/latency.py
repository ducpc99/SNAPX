import time

def measure_latency(model, data, batch_size=1):
    start_time = time.time()
    predictions, _ = model.batch_predict(batch_rows=data, llm_batch_size=batch_size)
    end_time = time.time()
    return end_time - start_time
