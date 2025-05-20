# modules/runner.py
from prompting import process_example

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