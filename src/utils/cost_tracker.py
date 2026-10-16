class CostTracker:
    def __init__(self):
        self.records = []

    def add(self, model, input_tokens=0, output_tokens=0):
        self.records.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

    def summary(self):
        totals = {}
        for record in self.records:
            model = record["model"]
            if model not in totals:
                totals[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            totals[model]["calls"] += 1
            totals[model]["input_tokens"] += record["input_tokens"]
            totals[model]["output_tokens"] += record["output_tokens"]
        return totals
