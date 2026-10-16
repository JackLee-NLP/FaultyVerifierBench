from src.dataset.io_utils import read_jsonl, write_jsonl

def validate_solver(input_path='data/raw/solver_raw.jsonl', output_path='data/processed/solver_validated.jsonl'):
    items=[]
    for x in read_jsonl(input_path):
        if x.get('solver_response') and x.get('solver_correctness') in ['correct','incorrect','partially_correct']:
            x['solver_validation_status']='validated'; items.append(x)
    write_jsonl(items, output_path); return items
