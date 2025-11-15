from eval.explain_metrics import accuracy_at_1, accuracy_at_k, macro_f1, mrr, ndcg_at_k

def evaluate_performance(y_true, predictions, rankings, k_list=[1, 5]):
    metrics_out = {}
    metrics_out["acc@1"] = accuracy_at_1(y_true, predictions)
    for k in k_list:
        if k == 1:
            continue
        metrics_out[f"acc@{k}"] = accuracy_at_k(y_true, rankings, k=k)
        metrics_out[f"ndcg@{k}"] = ndcg_at_k(y_true, rankings, k=k)
    metrics_out["macro_f1"] = macro_f1(y_true, predictions)
    metrics_out["mrr"] = mrr(y_true, rankings)
    return metrics_out
