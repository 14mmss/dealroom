import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import re

dataset = pd.read_csv('data_scientist_intern_revenue_model.csv')

rev_2017 = dataset['Revenue_2017'].replace(',', '', regex=True).fillna(0)
rev_2017_list = []
for x in rev_2017:
    y = int(re.sub("[^0-9]", '0', str(x)))
    rev_2017_list.append(y)

rev_2018 = dataset['Revenue_2018'].replace(',', '', regex=True).fillna(0)
rev_2018_list = []
for x in rev_2018:
    y = int(re.sub("[^0-9]", '0', str(x)))
    rev_2018_list.append(y)

rev_2019 = dataset['Revenue_2019'].replace(',', '', regex=True).fillna(0)
rev_2019_list = []
for x in rev_2019:
    y = int(re.sub("[^0-9]", '0', str(x)))
    rev_2019_list.append(y)

emp_2017 = dataset['Employees_2017'].fillna(0)
emp_2017_list = []
for i in emp_2017:
    emp_2017_list.append(i)

emp_2018 = dataset['Employees_2018'].fillna(0)
emp_2018_list = []
for i in emp_2018:
    emp_2018_list.append(i)

emp_2019 = dataset['Employees_2019'].fillna(0)
emp_2019_list = []
for i in emp_2019:
    emp_2019_list.append(i)

emp_2020 = dataset['Employees_2020'].fillna(0)
emp_2020_list = []
for i in emp_2020:
    emp_2020_list.append(i)

# create new clean dataset
df = pd.DataFrame(
    list(zip(rev_2017_list, rev_2018_list, rev_2019_list, emp_2017_list, emp_2018_list, emp_2019_list, emp_2020_list)),
    columns=['Revenue_2017', 'Revenue_2018', 'Revenue_2019', 'Employees_2017', 'Employees_2018', 'Employees_2019',
             'Employees_2020'])

x = df.iloc[:, [3, 4, 5]].values  # employee numbers
y = df.iloc[:, [0, 1, 2]].values  # revenue values

# split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# format data
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# begin regression
forest = RandomForestRegressor(n_estimators=1000, random_state=42)
forest.fit(x_train, y_train)

# Predicting the test set result
y_pred = forest.predict(x_test)
print(y_pred)

np.savetxt('predictions.csv', y_pred, delimiter=',')
