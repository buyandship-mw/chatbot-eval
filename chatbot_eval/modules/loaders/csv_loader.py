
import os
import csv
from typing import List
from modules.data import DataItem, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class CSVDataLoader(DataLoader):
    def __init__(self,
                 data_dir: str = "data",
                 text_col: str = "modified_conversation_body",
                 label_col: str = "ai_issue_tag",
                 pass_fail_col: str = "ai_performance"):  # New column for pass/fail
        self.base = os.path.join(PROJECT_ROOT, data_dir)
        self.text_col = text_col
        self.label_col = label_col
        self.pass_fail_col = pass_fail_col  # New attribute for pass/fail

    def load(self, filename: str) -> List[DataItem]:
        path = os.path.join(self.base, filename)
        items: List[DataItem] = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(self.text_col)
                raw = row.get(self.label_col)
                pass_fail = row.get(self.pass_fail_col)
                
                if raw and raw.strip():
                    tags = [t.strip() for t in raw.split(',') if t.strip()]
                else:
                    tags = []
                
                # Create DataItem with pass_fail field
                items.append(DataItem(text=text, expected=tags, pass_fail=pass_fail))
        return items