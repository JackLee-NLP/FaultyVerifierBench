from src.evaluation.score_final_answer import score_final_answer

def test_score_final_answer():
    exact = score_final_answer('24','24','math_reasoning','')
    assert exact["correct"]

    contradiction = score_final_answer('yes no','yes','factual_qa','')
    assert contradiction["correct"]
    assert contradiction["possible_negation_conflict"]
    assert contradiction["extra_prediction_tokens"] == ["no"]

    reversed_tokens = score_final_answer('beta alpha','alpha beta','factual_qa','')
    assert reversed_tokens["correct"]
    assert reversed_tokens["material_order_reversal"]
