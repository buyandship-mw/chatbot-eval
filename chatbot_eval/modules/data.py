import os
import csv
from collections import Counter

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

@dataclass
class ConvoItem:
    text: str
    pass_fail: Optional[str] = None
    expected: Optional[str] = None

class ConvoLoader(ABC):
    @abstractmethod
    def load(self, filename: str) -> List[ConvoItem]:
        """
        Load items from `<data_dir>/<filename>` into a list of ConvoItem.
        """
        ...

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

def print_failure_distribution(data) -> None:
    """
    Prints the distribution of pass/fail statuses in the dataset in descending order.
    
    Expects the data to be a list of ConvoItem objects with a 'pass_fail' attribute.
    """
    pass_fail_counter = Counter()
    for item in data:
        pass_fail_counter.update([item.pass_fail])  # Count occurrences of pass/fail

    pass_fail_counter = pass_fail_counter.most_common()

    print("Pass/Fail distribution in dataset (sorted by count):")
    for status, count in pass_fail_counter:
        print(f"{status}: {count}")

def print_hashtag_distribution(data: List, valid_tags: List[str]) -> None:
    """
    Prints the distribution of valid hashtags in the dataset in descending order,
    including tags with zero occurrences.
    
    Parameters:
        data (list): List of ConvoItem objects, each with an 'expected' list of hashtags.
        valid_tags (list): The full list of tags we want to track.
    """
    # Initialize counts for every valid tag
    counts = {tag: 0 for tag in valid_tags}

    # Count only valid tags
    for item in data:
        tags = item.expected or []
        for tag in tags:
            if tag in counts:
                counts[tag] += 1

    # Sort tags by count descending, then name
    sorted_counts = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0])
    )

    print("Hashtag distribution in dataset (sorted by count):")
    for tag, count in sorted_counts:
        print(f"{tag}: {count}")

# if __name__ == "__main__":
#     # Load the data
#     data_train, data_test = load_data()

#     # Get the list of valid hashtags
#     tags = get_tags()
#     print(f"Valid hashtags: {tags}\n")

#     # Print the distribution of hashtags in the training set
#     print_hashtag_distribution(data_train)

