
import os
import csv
from modules.data import DataItem, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class CSVDataLoader(DataLoader):
    def __init__(self, data_dir: str = "data", train_file: str = "dataset-train.csv", test_file: str = "dataset-test.csv"):
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)
        self.train_file = train_file
        self.test_file = test_file

    def load_data(self):
        train_path = os.path.join(self.data_dir, self.train_file)
        test_path  = os.path.join(self.data_dir, self.test_file)

        # Define CSV column -> DataItem attribute mapping
        COLUMN_MAP = {
            'modified_conversation_body': 'text',   # CSV column 'message' maps to DataItem 'text'
            'ai_issue_tag': 'expected', # CSV column 'label' maps to DataItem 'expected'
        }

        train = [self.row_to_dataitem(row, COLUMN_MAP) for row in self._read_csv(train_path)]
        test  = [self.row_to_dataitem(row, COLUMN_MAP) for row in self._read_csv(test_path)]
        
        print(f"Train: {len(train)}  Test: {len(test)}  Example: {train[0]}")
        return train, test

    def _read_csv(self, path):
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def row_to_dataitem(row, column_map):
        kwargs = {}
        for csv_col, dataitem_attr in column_map.items():
            value = row.get(csv_col, None)
            # Special logic for 'expected' (tags/labels)
            if dataitem_attr == 'expected':
                if value and value.strip():
                    # Split by comma, strip whitespace, filter empty
                    tags = [tag.strip() for tag in value.split(',') if tag.strip()]
                    kwargs[dataitem_attr] = tags
                else:
                    kwargs[dataitem_attr] = []
            else:
                kwargs[dataitem_attr] = value
        return DataItem(**kwargs)
