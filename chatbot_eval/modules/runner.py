# modules/runner.py
from modules.openai_client import prompt_model
from modules.evaluator     import construct_prompt, extract_valid_hashtags

def process_example(idx, test_data, demos_text, tags):
    """
    Run one test example through the model.
    Returns (result_dict, None) on success, (None, error_dict) on failure.
    """
    prompt = construct_prompt(demos_text, test_data["text"], tags)
    try:
        response = prompt_model(prompt)
        print(f"{idx}: {response}")
        predicted = extract_valid_hashtags(response, tags)
        return {
            "review":      test_data["text"],
            "true_labels": test_data["expected"],
            "predicted":   predicted,
        }, None
    except Exception as e:
        return None, {
            "review":      test_data["text"],
            "true_labels": test_data["expected"],
            "error":       str(e),
        }

def run_tests(data_test, demos_text, tags):
    """
    Iterate over all test examples, collect successes/errors.
    """
    results, errors = [], []
    for idx, test_data in enumerate(data_test):
        res, err = process_example(idx, test_data, demos_text, tags)
        if res:    results.append(res)
        if err:    errors.append(err)
    return results, errors