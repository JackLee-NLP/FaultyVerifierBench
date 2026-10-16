def classify_error(item, parsed_output, scores):
    if not scores.get('verifier_judgment_correct') and item.get('verifier_correctness') == 'faulty':
        return 'over_acceptance_or_faulty_verifier_failure'
    if not scores.get('verifier_judgment_correct') and item.get('verifier_correctness') == 'correct':
        return 'over_rejection_or_valid_verifier_failure'
    if scores.get('verifier_judgment_correct') and not scores.get('final_answer_correct'):
        return 'correct_judgment_wrong_answer'
    if scores.get('final_answer_correct') and not scores.get('verifier_judgment_correct'):
        return 'correct_answer_wrong_judgment'
    return 'other'
