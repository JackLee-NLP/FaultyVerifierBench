REQUIRED_BENCHMARK_FIELDS = ['id','question','gold_answer','solver_response','verifier_response']

def has_required_fields(item, required=REQUIRED_BENCHMARK_FIELDS):
    return all(k in item and item[k] not in [None,''] for k in required)
