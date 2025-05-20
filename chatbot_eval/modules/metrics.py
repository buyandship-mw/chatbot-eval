from modules.io import save_to_json


def evaluator_metrics(results):
    """
    Evaluates the results of the experiment by calculating overall precision, recall, and F1 score.
    Metrics are calculated per instance and then averaged across all results.
    
    Parameters:
        results (list): The list of results. Each result object should contain a list of true and a list of predicted labels.
        
    Returns:
        dict: A dictionary containing overall precision, recall, and F1 score.
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for result in results:
        true_labels = result["true_labels"]
        predicted_labels = result["predicted"]
        
        # Count true positives, false positives, and false negatives
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        # Count occurrences in true and predicted lists
        true_counts = {}
        predicted_counts = {}

        for label in true_labels:
            true_counts[label] = true_counts.get(label, 0) + 1

        for label in predicted_labels:
            predicted_counts[label] = predicted_counts.get(label, 0) + 1

        # Calculate true positives
        for label in predicted_counts:
            if label in true_counts:
                tp += min(predicted_counts[label], true_counts[label])

        # Calculate false positives
        for label in predicted_counts:
            if label not in true_counts:
                fp += predicted_counts[label]
            else:
                fp += max(0, predicted_counts[label] - true_counts[label])

        # Calculate false negatives
        for label in true_counts:
            if label not in predicted_counts:
                fn += true_counts[label]
            else:
                fn += max(0, true_counts[label] - predicted_counts[label])

        # Precision, recall, and F1 for this result  
        if not true_labels and not predicted_labels: # If both empty -> perfect match
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0


        # Accumulate metrics
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    # Calculate averages
    n = len(results)
    res = {
        "precision": total_precision / n,
        "recall": total_recall / n,
        "f1": total_f1 / n,
    }
    save_to_json("evaluator_metrics.json", res)
    return res

def chatbot_metrics(results):
    pass