# modules/reporting.py
def print_metrics(metrics):
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize():<9}: {value:.3f}")
