import torch

def measure_resources(model, data, batch_size=1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device)
        model.batch_predict(batch_rows=data, llm_batch_size=batch_size)
        final_memory = torch.cuda.memory_allocated(device)
        return final_memory - initial_memory
    return None
