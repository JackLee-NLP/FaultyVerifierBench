from src.dataset.schemas import QuestionItem

def test_question_schema():
    q=QuestionItem(id='1',question='Q',gold_answer='A',gold_explanation='E',domain='math',difficulty='easy')
    assert q.id=='1'
