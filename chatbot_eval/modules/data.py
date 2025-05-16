import os
import csv
from collections import Counter

from modules.io import read_from_json

def load_data() -> tuple:
    """
    Load the dataset from JSON files.
    """
    # Define the path to the data directory
    config_file = os.path.join(os.path.dirname(__file__), "../data/")

    # Load training data
    data_train = read_from_json(os.path.join(config_file, 'dataset-train.json'))
    data_test = read_from_json(os.path.join(config_file, 'dataset-dev.json'))

    print('\nDataset Sizes: Train %i, Test %i\n' % (len(data_train), len(data_test)))
    print(data_train[0] + "\n")

    return data_train, data_test

# Could modify this to extract tags from training data instead
def get_tags(csv_filename: str = "valid_tags.csv") -> list:
    """
    Load the list of valid hashtags from a CSV file.
    The CSV file should contain one hashtag per line.
    """
    # build a path to tags.csv in the same directory as this script
    base_dir = os.path.join(os.path.dirname(__file__), "../config/")
    path = os.path.join(base_dir, csv_filename)

    tags = []
    # open as a simple one-column CSV (no header)
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # skip empty lines
            if not row:
                continue
            # take the first column and strip whitespace/newlines
            tags.append(row[0].strip())
    return tags

def print_hashtag_distribution(data) -> None:
    """
    Prints the distribution of hashtags in the dataset in descending order.
    
    Expects the data to be a list of dictionaries with an 'expected' key.
    Each dictionary should contain a list of hashtags under the 'expected' key.
    """
    hashtag_counter = Counter()
    for item in data:
        hashtag_counter.update(item['expected'])
    hashtag_counter = hashtag_counter.most_common()

    print("Hashtag distribution in training set (sorted by count):")
    for tag, count in hashtag_counter:
        print(f"{tag}: {count}")

if __name__ == "__main__":
    # Load the data
    data_train, data_test = load_data()

    # Get the list of valid hashtags
    tags = get_tags()
    print(f"Valid hashtags: {tags}\n")

    # Print the distribution of hashtags in the training set
    print_hashtag_distribution(data_train)

