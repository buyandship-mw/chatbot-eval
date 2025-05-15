from modules.openai_client import prompt_model
from modules.io import read_from_json, save_to_json
from modules.data import load_data, get_tags, print_hashtag_distribution, sample_demonstrations
from modules.evaluator import linearize_demonstrations, construct_prompt, extract_valid_hashtags
from modules.metrics import evaluate_results

def main():
    # Load the data
    data_train, data_test = load_data()

    # Get the list of valid hashtags
    tags = get_tags()
    print(f"Valid hashtags: {tags}\n")

    # Print the distribution of hashtags in the training set
    print_hashtag_distribution(data_train)
    print("\n")

    # Run the experiment
    results = []
    errors = []

    # Sample and linearize demonstrations
    demonstrations = sample_demonstrations(data_train)
    demonstrations_text = linearize_demonstrations(demonstrations)
        
    for idx, test_data in enumerate(data_test):
        # Construct the full prompt and get response from the model
        prompt = construct_prompt(demonstrations_text, test_data['text'], tags)
        try:
            response = prompt_model(prompt)
            print(f"{idx}: {response}")
            
            # Remove invalid tags from the response
            predicted_categories = extract_valid_hashtags(response, tags)
            results.append({
                "review": test_data["text"],
                "true_labels": test_data["expected"],
                "predicted": predicted_categories,
            })
        except Exception as e:
            # Catch and log any other errors
            errors.append({
                "review": test_data["text"],
                "true_labels": test_data["expected"],
                "error": str(e),
            })

    # Save the results and errors to files
    save_to_json("results.json", results)
    save_to_json("errors.json", errors)

    # Print summary
    print(f"Experiment completed. {len(results)} results saved to 'results.json'.\n")

    # Evaluate the results
    results = read_from_json("results.json")
    metrics = evaluate_results(results)

    # Print the overall metrics
    print("Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")

if __name__ == "__main__":
    main()