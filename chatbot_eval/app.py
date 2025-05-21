from modules.io        import read_from_json, save_to_json
from modules.data      import get_tags, print_failure_distribution, print_hashtag_distribution
from modules.loaders.csv_loader import CSVConvoLoader
from modules.sampling import sample_demonstrations
from modules.runner    import run_tests
from modules.metrics   import evaluator_metrics, chatbot_metrics, print_evaluator_metrics, print_chatbot_metrics

def main():
    # Load data
    loader = CSVConvoLoader()
    demos = loader.load("demo.csv")
    test_data = loader.load("test.csv")

    # Setup
    demos = sample_demonstrations(demos)
    test_data = test_data[:10]
    print(f"Loaded {len(demos)} demonstrations and {len(test_data)} test examples.")
    print(f"Example: {demos[0]}\n")
    
    tags = get_tags()
    print(f"Valid hashtags: {tags}\n")

    print("Analyzing demonstrations data...")
    print_failure_distribution(demos)
    print_hashtag_distribution(demos, tags)
    print()

    # Test evaluator
    run_tests(test_data, demos, tags)

    # Report evaluator metrics
    results = read_from_json("results.json")
    metrics = evaluator_metrics(results)
    print_evaluator_metrics(metrics)
    print()
    chatbot_met = chatbot_metrics(results)
    print_chatbot_metrics(chatbot_met)

if __name__ == "__main__":
    main()