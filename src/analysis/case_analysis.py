from collections import defaultdict

def summarize_by_case(items, score_rows):
    scores={r['id']:r for r in score_rows}; groups=defaultdict(list)
    for x in items:
        key=(x.get('solver_correctness'), x.get('verifier_correctness'))
        if x['id'] in scores: groups[key].append(scores[x['id']])
    return {str(k): sum(r['final_answer_correct'] for r in v)/len(v) for k,v in groups.items() if v}
