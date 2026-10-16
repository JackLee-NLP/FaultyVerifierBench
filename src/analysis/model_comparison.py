def compare_models(score_dicts):
    return sorted(score_dicts, key=lambda x: x.get('final_answer_accuracy',0), reverse=True)
