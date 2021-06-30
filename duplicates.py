import pandas as pd

# read csv
df = pd.read_csv('data_scientist_duplicate_detection.csv')

# find duplicate rows
duplicate_rows = df[df.duplicated()]

# show rows
print("\n\nDuplicate Rows : \n {}".format(duplicate_rows))

# save them to csv file
duplicate_rows.to_csv("duplicates.csv")
