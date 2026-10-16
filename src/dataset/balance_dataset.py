from collections import defaultdict
from src.dataset.io_utils import read_jsonl, write_jsonl

def balance_dataset(input_path='data/processed/benchmark_dataset.jsonl', output_path='data/processed/benchmark_dataset_balanced.jsonl', max_per_group=100):
    groups=defaultdict(list)
    for x in read_jsonl(input_path):
        key=(x.get('domain'), x.get('solver_correctness'), x.get('verifier_correctness'))
        if len(groups[key]) < max_per_group: groups[key].append(x)
    items=[i for g in groups.values() for i in g]
    write_jsonl(items, output_path); return items
