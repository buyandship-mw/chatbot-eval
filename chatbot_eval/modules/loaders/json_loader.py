# modules/data.py (continued)

import os
from typing import List
from modules.data import DataItem, DataLoader
from modules.io import read_from_json  # update this import as needed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class JSONDataLoader(DataLoader):
    def __init__(self, data_dir: str = "data"):
        self.base = os.path.join(PROJECT_ROOT, data_dir)

    def load(self, filename: str) -> List[DataItem]:
        path = os.path.join(self.base, filename)
        raw = read_from_json(path)
        # drop any extra keys
        allowed = {'text', 'expected'}
        return [
            DataItem(**{k: v for k, v in entry.items() if k in allowed})
            for entry in raw
        ]