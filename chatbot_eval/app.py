from modules.io        import read_from_json, save_to_json
from modules.data      import get_tags, print_failure_distribution, print_hashtag_distribution
from modules.loaders.csv_loader import CSVDataLoader
from modules.sampling import sample_demonstrations
from modules.runner    import run_tests
from modules.metrics   import evaluator_metrics, print_evaluator_metrics

def main():
    # Load data
    loader = CSVDataLoader()
    demos = loader.load("demos.csv")
    test_data = loader.load("testset.csv")

    # Setup
    demos = sample_demonstrations(demos)
    test_data = test_data[:10]
    print(f"Loaded {len(demos)} demonstrations and {len(test_data)} test examples.")
    print(f"Example: {demos[0]}\n")
    
    tags = get_tags()
    print(f"Valid hashtags: {tags}\n")

    print("Distribution of demonstrations in dataset:")
    print_failure_distribution(demos)
    print_hashtag_distribution(demos, tags)

    # Test evaluator
    # run_tests(test_data, demos, tags)

    # Report evaluator metrics
    results = read_from_json("results.json")
    metrics = evaluator_metrics(results)
    print_evaluator_metrics(metrics)

if __name__ == "__main__":
    main()