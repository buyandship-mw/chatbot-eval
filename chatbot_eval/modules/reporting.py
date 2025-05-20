# reporting.py
def print_metrics(metrics):
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