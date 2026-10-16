import pandas as pd

def export_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
