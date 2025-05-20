from modules.io import save_to_json

def evaluator_metrics(results):
    """
    Evaluates the results of the experiment by calculating precision, recall, and F1 score for:
      - Pass/Fail classification (macro-averaged over all examples)
      - Tag classification (macro-averaged over failure examples; perfect if no failures)
    
    Parameters:
        results (list): Each result is a dict with keys:
            "pass_fail"            : str, ground-truth "Pass" or "Fail"
            "predicted_pass_fail"  : str, predicted "Pass"/"Fail"/"Unknown"
            "tags"                 : list[str], ground-truth tags (empty list if none)
            "predicted_tags"       : list[str], predicted tags
        
    Returns:
        dict: {
            "pass_fail_metrics": {"precision":…, "recall":…, "f1":…},
            "tag_metrics"      : {"precision":…, "recall":…, "f1":…}
        }
    """
    total_pf_p = total_pf_r = total_pf_f1 = 0.0
    total_tag_p = total_tag_r = total_tag_f1 = 0.0
    num_examples = len(results)
    num_failures = 0

    for result in results:
        # --- Pass/Fail metrics (macro over all) ---
        true_pf = result["pass_fail"]
        pred_pf = result["predicted_pass_fail"]

        tp_pf = 1 if pred_pf == true_pf else 0
        fp_pf = 1 if (pred_pf == "Fail" and true_pf != "Fail") else 0
        fn_pf = 1 if (pred_pf == "Pass" and true_pf != "Pass") else 0

        # precision / recall / f1 for this example
        p_pf = tp_pf / (tp_pf + fp_pf) if (tp_pf + fp_pf) > 0 else 0.0
        r_pf = tp_pf / (tp_pf + fn_pf) if (tp_pf + fn_pf) > 0 else 0.0
        f1_pf = (2 * tp_pf) / (2 * tp_pf + fp_pf + fn_pf) if (2 * tp_pf + fp_pf + fn_pf) > 0 else 0.0

        total_pf_p  += p_pf
        total_pf_r  += r_pf
        total_pf_f1 += f1_pf

        # --- Tag metrics only for failures ---
        if true_pf == "Fail":
            num_failures += 1
            true_tags      = result["tags"]
            predicted_tags = result["predicted_tags"]

            # count TP/FP/FN
            tp_t = fp_t = fn_t = 0
            true_counts = {}
            pred_counts = {}
            for t in true_tags:      true_counts[t] = true_counts.get(t, 0) + 1
            for t in predicted_tags: pred_counts[t] = pred_counts.get(t, 0) + 1

            for tag, cnt in pred_counts.items():
                if tag in true_counts:
                    tp_t += min(cnt, true_counts[tag])
                    fp_t += max(0, cnt - true_counts[tag])
                else:
                    fp_t += cnt

            for tag, cnt in true_counts.items():
                if tag not in pred_counts:
                    fn_t += cnt
                else:
                    fn_t += max(0, cnt - pred_counts[tag])

            p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
            r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            f1_t = (2 * tp_t) / (2 * tp_t + fp_t + fn_t) if (2 * tp_t + fp_t + fn_t) > 0 else 0.0

            total_tag_p  += p_t
            total_tag_r  += r_t
            total_tag_f1 += f1_t

    # Average pass/fail over all examples
    pf_metrics = {
        "precision": total_pf_p  / num_examples if num_examples > 0 else 0.0,
        "recall"   : total_pf_r  / num_examples if num_examples > 0 else 0.0,
        "f1"       : total_pf_f1 / num_examples if num_examples > 0 else 0.0,
    }

    # Average tag metrics over failures only (or perfect if none)
    if num_failures > 0:
        tag_metrics = {
            "precision": total_tag_p  / num_failures,
            "recall"   : total_tag_r  / num_failures,
            "f1"       : total_tag_f1 / num_failures,
        }
    else:
        tag_metrics = {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Persist to disk
    save_to_json("pass_fail_metrics.json", pf_metrics)
    save_to_json("tag_metrics.json", tag_metrics)

    return {
        "pass_fail_metrics": pf_metrics,
        "tag_metrics"      : tag_metrics,
    }

def print_evaluator_metrics(metrics):
    """
    Prints the precision, recall, and F1 scores for both pass/fail and tags.
    
    Parameters:
        metrics (dict): The dictionary containing the pass/fail and tag classification metrics.
    """
    # Print pass/fail metrics
    print("Pass/Fail Metrics:")
    for metric, value in metrics["pass_fail_metrics"].items():
        print(f"{metric.capitalize():<9}: {value:.3f}")
    
    # Print tag classification metrics
    print("\nTag Classification Metrics:")
    for metric, value in metrics["tag_metrics"].items():
        print(f"{metric.capitalize():<9}: {value:.3f}")

def chatbot_metrics(results):
    pass