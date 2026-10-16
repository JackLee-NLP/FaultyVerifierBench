from src.dataset.io_utils import read_jsonl, write_jsonl

def validate_questions(input_path='data/raw/questions_raw.jsonl', output_path='data/processed/questions_validated.jsonl'):
    items=[]
    for x in read_jsonl(input_path):
        x['validation_status']='validated' if x.get('question') and x.get('gold_answer') else 'invalid'
        if x['validation_status']=='validated': items.append(x)
    write_jsonl(items, output_path); return items
