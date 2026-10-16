from collections import defaultdict

def summarize_by_domain(items, score_rows):
    scores={r['id']:r for r in score_rows}; groups=defaultdict(list)
    for x in items:
        if x['id'] in scores: groups[x.get('domain')].append(scores[x['id']])
    return {k: sum(r['final_answer_correct'] for r in v)/len(v) for k,v in groups.items() if v}
