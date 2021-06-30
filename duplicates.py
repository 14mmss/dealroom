import pandas as pd

df = pd.read_csv('data_scientist_duplicate_detection.csv')

duplicate_rows = df[df.duplicated()]

print("\n\nDuplicate Rows : \n {}".format(duplicate_rows))

duplicate_rows.to_csv("duplicates.csv")
