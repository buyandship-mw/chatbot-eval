from collections import Counter
from modules.io import read_from_json, save_to_json

def evaluator_metrics(results, beta2=4.0):
    """
    Evaluates tag classification results by calculating precision, recall,
    and a weighted F-score (recall weighted β² as important as precision),
    macro-averaged over:
      1) all examples
      2) only the 'fail' examples (where there is at least one true tag)

    Parameters:
        results (list): Each result is a dict with keys:
            "tags"           : list[str], ground-truth tags (empty list if none)
            "predicted_tags" : list[str], predicted tags (empty list if none)
        beta2 (float):    β² value for F_β; default=4.0

    Returns:
        dict: {
            "precision"      : float, # over all examples
            "recall"         : float,
            "f_beta"         : float,
            "fail_precision" : float, # over only examples with tags
            "fail_recall"    : float,
            "fail_f_beta"    : float
        }
    """
    total_p = total_r = total_f = 0.0
    total_p_fail = total_r_fail = total_f_fail = 0.0
    n = len(results)
    n_fail = 0

    for result in results:
        true_tags = result["tags"]
        pred_tags = result["predicted_tags"]

        # build count maps
        true_counts = {}
        for t in true_tags:
            true_counts[t] = true_counts.get(t, 0) + 1
        pred_counts = {}
        for t in pred_tags:
            pred_counts[t] = pred_counts.get(t, 0) + 1

        # compute TP, FP, FN
        tp = fp = fn = 0
        for tag, cnt in pred_counts.items():
            if tag in true_counts:
                tp += min(cnt, true_counts[tag])
                fp += max(0, cnt - true_counts[tag])
            else:
                fp += cnt
        for tag, cnt in true_counts.items():
            if tag not in pred_counts:
                fn += cnt
            else:
                fn += max(0, cnt - pred_counts[tag])

        # edge‐cases: no true & no pred → perfect; no true tags → recall=1 if none missed
        if not true_counts and not pred_counts:
            p = r = 1.0
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        # F_β
        f = ((1 + beta2) * p * r) / (beta2 * p + r) if (beta2 * p + r) > 0 else 0.0

        # accumulate overall
        total_p += p
        total_r += r
        total_f += f

        # accumulate only failures (where there was at least one true tag)
        if true_tags:
            total_p_fail += p
            total_r_fail += r
            total_f_fail += f
            n_fail += 1

    # macro‐average overall
    avg_p = total_p / n if n > 0 else 0.0
    avg_r = total_r / n if n > 0 else 0.0
    avg_f = total_f / n if n > 0 else 0.0

    # macro‐average on fail subset
    avg_p_fail = total_p_fail / n_fail if n_fail > 0 else 0.0
    avg_r_fail = total_r_fail / n_fail if n_fail > 0 else 0.0
    avg_f_fail = total_f_fail / n_fail if n_fail > 0 else 0.0

    metrics = {
        "precision"      : avg_p,
        "recall"         : avg_r,
        "f_beta"         : avg_f,
        "fail_precision": avg_p_fail,
        "fail_recall"   : avg_r_fail,
        "fail_f_beta"   : avg_f_fail,
    }

    # Persist to disk if needed
    save_to_json("metrics_evaluator.json", metrics)

    return metrics

def chatbot_metrics(results):
    """
    Computes chatbot performance metrics, assuming all predictions are 'Pass' or 'Fail'.
    
    Parameters:
        results (list): Each result dict must contain:
            - "predicted_pass_fail": str, either "Pass" or "Fail"
            - "predicted_tags"     : list[str], tags assigned if it failed
    
    Returns:
        dict: {
            "total": int,
            "passes": int,
            "fails": int,
            "pass_rate": float,
            "fail_rate": float,
            "avg_tags_per_failure": float,
            "tag_distribution": { tag: count, ... }
        }
    """
    total   = len(results)
    passes  = sum(1 for r in results if r["predicted_pass_fail"] == "Pass")
    fails   = total - passes

    pass_rate = passes / total if total else 0.0
    fail_rate = fails  / total if total else 0.0

    # Average number of tags among failures
    tags_per_fail = [len(r["predicted_tags"]) for r in results if r["predicted_pass_fail"] == "Fail"]
    avg_tags = sum(tags_per_fail) / len(tags_per_fail) if tags_per_fail else 0.0

    # Distribution of predicted tags across all failures
    tag_counter = Counter()
    for r in results:
        if r["predicted_pass_fail"] == "Fail":
            tag_counter.update(r["predicted_tags"])

    metrics = {
        "total": total,
        "passes": passes,
        "fails": fails,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
        "avg_tags_per_failure": avg_tags,
        "tag_distribution": dict(tag_counter)
    }

    save_to_json("metrics_chatbot.json", metrics)
    return metrics

def print_evaluator_metrics(metrics):
    """
    Prints the precision, recall, and F1 scores for both pass/fail and tags.
    
    Parameters:
        metrics (dict): The dictionary containing the pass/fail and tag classification metrics.
    """
    print("Evaluator performance metrics:")
    # Print tag classification metrics
    for metric, value in metrics.items():
        print(f"{metric.capitalize():<9}: {value:.3f}")

def print_chatbot_metrics(metrics):
    """
    Nicely prints the chatbot performance metrics.
    """
    print("Chatbot performance metrics (per LLM):")
    print(f"Total convos       : {metrics['total']}")
    print(f"Passed             : {metrics['passes']}  ({metrics['pass_rate']:.3f})")
    print(f"Failed             : {metrics['fails']}  ({metrics['fail_rate']:.3f})")
    print(f"Avg tags per fail  : {metrics['avg_tags_per_failure']:.2f}\n")

    print("Failure tag distribution:")
    for tag, count in sorted(metrics["tag_distribution"].items(), key=lambda x: -x[1]):
        print(f"{tag}: {count}")

if __name__ == "__main__":
    results = read_from_json("results.json")
    metrics = evaluator_metrics(results)
    print_evaluator_metrics(metrics)
    print()
    chatbot_met = chatbot_metrics(results)
    print_chatbot_metrics(chatbot_met)