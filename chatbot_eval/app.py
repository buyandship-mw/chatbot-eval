from modules.io        import read_from_json, save_to_json
from modules.data      import get_tags, print_hashtag_distribution
from modules.loaders.csv_loader import CSVDataLoader
from modules.sampling import sample_demonstrations
from modules.prompting import linearize_demonstrations
from modules.runner    import run_tests
from modules.metrics   import evaluator_metrics
from modules.reporting import print_metrics

def main():
    # 1️⃣ Setup
    loader = CSVDataLoader()
    data_train = loader.load("dataset-train.csv")
    data_test = loader.load("dataset-test.csv")
    
    tags = get_tags()
    print(f"Valid hashtags: {tags}\n")
    print_hashtag_distribution(data_train)
    print(f"Train: {len(data_train)}  Test: {len(data_test)}\nExample: {data_train[0]}\n")

    # 2️⃣ Prepare demos
    demos = sample_demonstrations(data_train)
    demos_text = linearize_demonstrations(demos)

    # 3️⃣ Run the model on all test examples
    res, errors = run_tests(data_test, demos_text, tags)

    # 4️⃣ Persist
    save_to_json("results.json", res)
    save_to_json("errors.json", errors)
    print(f"\nExperiment completed. {len(res)} results saved to 'results.json'.\n")

    # 5️⃣ Report
    results = read_from_json("results.json")
    metrics = evaluator_metrics(results)
    print_metrics(metrics)

if __name__ == "__main__":
    main()