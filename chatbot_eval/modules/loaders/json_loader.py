# modules/data.py (continued)

import os
from modules.data import DataItem, DataLoader
from modules.io import read_from_json  # update this import as needed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print("PROJECT_ROOT:", PROJECT_ROOT)

class JSONDataLoader(DataLoader):
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)

    def load_data(self):
        raw_train = read_from_json(os.path.join(self.data_dir, 'dataset-train.json'))
        raw_test  = read_from_json(os.path.join(self.data_dir, 'dataset-dev.json'))

        ALLOWED_KEYS = {'text', 'expected'}
        train = [DataItem(**{k: v for k, v in d.items() if k in ALLOWED_KEYS}) for d in raw_train]
        test  = [DataItem(**{k: v for k, v in d.items() if k in ALLOWED_KEYS}) for d in raw_test]

        print(f"Train: {len(train)}  Test: {len(test)}\nExample: {train[0]}\n")
        return train, test