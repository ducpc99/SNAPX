def calculate_cpr(results, metrics):
    cpr_scores = {}
    for model_name, result in results.items():
        latency = result["latency"]
        memory = result["memory"]
        performance = metrics.get(f"{model_name}_performance", 0)
        cpr_scores[model_name] = performance / (latency + memory)
    return cpr_scores
