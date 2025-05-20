from modules.io import save_to_json

def evaluator_metrics(results):
    """
    Evaluates the results of the experiment by calculating overall precision, recall, and F1 score for:
    - Pass/Fail classification.
    - Tags classification for failed conversations.
    
    Metrics are calculated per instance and then averaged across all results.
    
    Parameters:
        results (list): The list of results. Each result object should contain a list of true and predicted labels.
        
    Returns:
        dict: A dictionary containing precision, recall, and F1 scores for both pass/fail and tag classification.
    """
    
    # Initialize metrics for pass/fail classification
    total_pass_fail_precision = 0
    total_pass_fail_recall = 0
    total_pass_fail_f1 = 0

    # Initialize metrics for tag classification
    total_tag_precision = 0
    total_tag_recall = 0
    total_tag_f1 = 0
    
    for result in results:
        # Extract true labels and predicted labels for pass/fail
        true_pass_fail = result["pass_fail"]
        predicted_pass_fail = result["predicted_pass_fail"]
        
        # Extract true tags and predicted tags for tag classification
        true_tags = result["tags"]
        predicted_tags = result["predicted_tags"]

        # --- Pass/Fail Classification Metrics ---
        tp_pass_fail = 0  # True positives for pass/fail
        fp_pass_fail = 0  # False positives for pass/fail
        fn_pass_fail = 0  # False negatives for pass/fail

        if predicted_pass_fail == true_pass_fail:
            tp_pass_fail = 1
        else:
            if true_pass_fail == "Pass":
                fn_pass_fail = 1
            elif true_pass_fail == "Fail":
                fp_pass_fail = 1
        
        # Precision, recall, and F1 for pass/fail
        precision_pass_fail = tp_pass_fail / (tp_pass_fail + fp_pass_fail) if (tp_pass_fail + fp_pass_fail) > 0 else 0
        recall_pass_fail = tp_pass_fail / (tp_pass_fail + fn_pass_fail) if (tp_pass_fail + fn_pass_fail) > 0 else 0
        f1_pass_fail = (2 * tp_pass_fail) / (2 * tp_pass_fail + fp_pass_fail + fn_pass_fail) if (2 * tp_pass_fail + fp_pass_fail + fn_pass_fail) > 0 else 0
        
        total_pass_fail_precision += precision_pass_fail
        total_pass_fail_recall += recall_pass_fail
        total_pass_fail_f1 += f1_pass_fail

        # --- Tag Classification Metrics (only for failed conversations) ---
        if true_pass_fail == "Fail":
            # Count true positives, false positives, and false negatives for tags
            tp_tags = 0
            fp_tags = 0
            fn_tags = 0
            
            true_counts = {}
            predicted_counts = {}
            
            # Count occurrences in true and predicted tag lists
            for tag in true_tags:
                true_counts[tag] = true_counts.get(tag, 0) + 1
            for tag in predicted_tags:
                predicted_counts[tag] = predicted_counts.get(tag, 0) + 1

            # Calculate true positives
            for tag in predicted_counts:
                if tag in true_counts:
                    tp_tags += min(predicted_counts[tag], true_counts[tag])

            # Calculate false positives
            for tag in predicted_counts:
                if tag not in true_counts:
                    fp_tags += predicted_counts[tag]
                else:
                    fp_tags += max(0, predicted_counts[tag] - true_counts[tag])

            # Calculate false negatives
            for tag in true_counts:
                if tag not in predicted_counts:
                    fn_tags += true_counts[tag]
                else:
                    fn_tags += max(0, true_counts[tag] - predicted_counts[tag])

            # Precision, recall, and F1 for tag classification
            precision_tags = tp_tags / (tp_tags + fp_tags) if (tp_tags + fp_tags) > 0 else 0
            recall_tags = tp_tags / (tp_tags + fn_tags) if (tp_tags + fn_tags) > 0 else 0
            f1_tags = (2 * tp_tags) / (2 * tp_tags + fp_tags + fn_tags) if (2 * tp_tags + fp_tags + fn_tags) > 0 else 0

            total_tag_precision += precision_tags
            total_tag_recall += recall_tags
            total_tag_f1 += f1_tags
    
    # Calculate averages for pass/fail classification
    n = len(results)
    pass_fail_metrics = {
        "precision": total_pass_fail_precision / n,
        "recall": total_pass_fail_recall / n,
        "f1": total_pass_fail_f1 / n,
    }
    
    # Calculate averages for tag classification (failures only)
    tag_metrics = {
        "precision": total_tag_precision / n,
        "recall": total_tag_recall / n,
        "f1": total_tag_f1 / n,
    }

    # Save to JSON
    save_to_json("pass_fail_metrics.json", pass_fail_metrics)
    save_to_json("tag_metrics.json", tag_metrics)

    return {
        "pass_fail_metrics": pass_fail_metrics,
        "tag_metrics": tag_metrics,
    }

def chatbot_metrics(results):
    pass