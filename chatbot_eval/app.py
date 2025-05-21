from modules.io        import read_from_json, save_to_json
from modules.data      import load_tag_definitions, print_failure_distribution, print_hashtag_distribution
from modules.loaders.csv_loader import CSVDataLoader
from modules.sampling import sample_demonstrations
from modules.runner    import run_tests
from modules.metrics   import evaluator_metrics, chatbot_metrics, print_evaluator_metrics, print_chatbot_metrics

def main():
    # Load the data
    loader = CSVDataLoader()
    demos = loader.load("demo.csv")
    demos = sample_demonstrations(demos)
    test_data = loader.load("test.csv")
    test_data = test_data[:20]

    print(f"Loaded {len(demos)} demonstrations and {len(test_data)} test examples.")
    print(f"Example: {demos[0]}\n")

    tags = load_tag_definitions("valid_tags.csv")
    print("Valid hashtags and their descriptions:")
    for tag, desc in tags.items():
        print(f"{tag}: {desc}")
    print()

    print("Analyzing demonstrations data...")
    print_failure_distribution(demos)
    print_hashtag_distribution(demos, list(tags.keys()))
    print()

    print("Analyzing test data...")
    print_failure_distribution(test_data)
    print_hashtag_distribution(test_data, list(tags.keys()))
    print()

    print("Running evaluator on test data...")
    run_tests(test_data, demos, tags)

    results = read_from_json("results.json")
    metrics = evaluator_metrics(results)
    print_evaluator_metrics(metrics)
    print()
    chatbot_met = chatbot_metrics(results)
    print_chatbot_metrics(chatbot_met)

if __name__ == "__main__":
    main()