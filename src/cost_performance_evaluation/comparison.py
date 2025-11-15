def compare_models(cfg, data, batch_size=1):
    results = {}
    models = ['hybrid', 'sequence_only', 'llm_only']
    for model_name in models:
        model = load_model(model_name, cfg)  # Giả sử có một hàm load_model() để tải mô hình
        latency = measure_latency(model, data, batch_size)
        memory_usage = measure_resources(model, data, batch_size)
        results[model_name] = {"latency": latency, "memory": memory_usage}
    return results
