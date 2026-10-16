import pandas as pd
from src.dataset.io_utils import read_jsonl

def export_for_human_review(input_path='data/processed/benchmark_dataset.jsonl', output_path='data/processed/human_review_items.csv', n=100):
    items=read_jsonl(input_path)[:n]
    pd.DataFrame(items).to_csv(output_path, index=False)
