from prompts.adjudication_prompts import build_adjudication_prompt

def test_prompt_format():
    item={'question':'Q','solver_response':'S','verifier_response':'V'}
    s,u=build_adjudication_prompt(item)
    assert 'Original question' in u
