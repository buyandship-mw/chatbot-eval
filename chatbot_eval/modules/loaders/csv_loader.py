
import os
import csv
from typing import List
from modules.data import DataItem, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class CSVDataLoader(DataLoader):
    def __init__(self,
                 data_dir: str = "data",
                 text_col: str = "modified_conversation_body",
                 label_col: str = "ai_issue_tag"):
        self.base     = os.path.join(PROJECT_ROOT, data_dir)
        self.text_col = text_col
        self.label_col= label_col

    def load(self, filename: str) -> List[DataItem]:
        path = os.path.join(self.base, filename)
        items: List[DataItem] = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(self.text_col, "")
                raw = row.get(self.label_col, "")
                if raw and raw.strip():
                    tags = [t.strip() for t in raw.split(',') if t.strip()]
                else:
                    tags = []
                items.append(DataItem(text=text, expected=tags))
        return items