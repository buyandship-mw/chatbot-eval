from modules.io        import save_to_json
from modules.data      import get_tags, print_hashtag_distribution
from modules.loaders.json_loader import JSONDataLoader
from modules.loaders.csv_loader import CSVDataLoader
from modules.sampling import sample_demonstrations
from modules.prompting import linearize_demonstrations
from modules.runner    import run_tests
from modules.metrics   import evaluate_results
from modules.reporting import print_summary, print_metrics

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
    results, errors = run_tests(data_test, demos_text, tags)

    # 4️⃣ Persist
    save_to_json("results.json", results)
    save_to_json("errors.json", errors)

    # 5️⃣ Report
    print_summary(len(results), "results.json")
    metrics = evaluate_results(results)
    print_metrics(metrics)

if __name__ == "__main__":
    main()