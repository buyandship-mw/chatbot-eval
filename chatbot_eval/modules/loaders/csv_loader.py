
import os
import csv
from typing import List
from modules.data import DataItem, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class CSVDataLoader(DataLoader):
    def __init__(self,
                 data_dir: str = "data",
                 text_col: str = "conversation_part",
                 label_col: str = "ai_issue_tag",
                 pass_fail_col: str = "ai_performance"):  # New column for pass/fail
        self.base = os.path.join(PROJECT_ROOT, data_dir)
        self.text_col = text_col
        self.label_col = label_col
        self.pass_fail_col = pass_fail_col  # New attribute for pass/fail

    from typing import List

    def load(self, filename: str) -> List[DataItem]:
        path = os.path.join(self.base, filename)
        items: List[DataItem] = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                line_no = reader.line_num

                # 1) Fetch text column and error if missing or blank
                text = row.get(self.text_col)
                if text is None:
                    raise KeyError(f"Required column '{self.text_col}' not found at line {line_no}")
                if not text.strip():
                    raise ValueError(f"Empty text value in column '{self.text_col}' at line {line_no}")

                # 2) Fetch and parse labels
                raw = row.get(self.label_col, '')
                if raw and raw.strip():
                    tags = [t.strip() for t in raw.split(',') if t.strip()]
                else:
                    tags = []

                # 3) Fetch pass/fail (allowing it to be None if column truly missing)
                pass_fail = row.get(self.pass_fail_col)

                items.append(DataItem(text=text, expected=tags, pass_fail=pass_fail))

        return items