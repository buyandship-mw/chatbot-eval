# modules/runner.py
from modules.io import save_to_json
from modules.prompting import linearize_demonstrations_pass_fail, linearize_demonstrations_tagging, process_example

def run_tests(data_test, demos, tags):
    """
    Iterate over all test examples, collect successes/errors.
    """
    pass_fail_demos_text = linearize_demonstrations_pass_fail(demos)
    tagging_demos_text = linearize_demonstrations_tagging(demos)

    results, errors = [], []
    for idx, test_data in enumerate(data_test):
        res, err = process_example(idx, test_data, pass_fail_demos_text, tagging_demos_text, tags)
        if res:    results.append(res)
        if err:    errors.append(err)

    save_to_json("results.json", results)
    save_to_json("errors.json", errors)
    
    print(f"\nExperiment completed. {len(results)} results saved to 'results.json'.\n")