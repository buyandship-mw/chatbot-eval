# modules/reporting.py
def print_summary(results_count, results_path="results.json"):
    print(f"\nExperiment completed. {results_count} results saved to '{results_path}'.\n")

def print_metrics(metrics):
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize():<9}: {value:.3f}")
