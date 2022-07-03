import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier


lgbm = LGBMClassifier()

df = pd.read_csv('C:/Users/manas/Downloads/Census Income/adult.csv')
df.columns=df.columns.str.strip()
#print(df.head())
print(df.describe())
'''
                age        fnlwgt  ...  capital.loss  hours.per.week
count  32561.000000  3.256100e+04  ...  32561.000000    32561.000000
mean      38.581647  1.897784e+05  ...     87.303830       40.437456
std       13.640433  1.055500e+05  ...    402.960219       12.347429
min       17.000000  1.228500e+04  ...      0.000000        1.000000
25%       28.000000  1.178270e+05  ...      0.000000       40.000000
50%       37.000000  1.783560e+05  ...      0.000000       40.000000
75%       48.000000  2.370510e+05  ...      0.000000       45.000000
max       90.000000  1.484705e+06  ...   4356.000000       99.000000

[8 rows x 6 columns]

'''
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
              'race', 'sex', 'capita_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.income = np.where(df['income'] == ' >50K', 1, 0)

df['sex'] = np.where(df['sex'] == ' Male', 1, 0)
df['native_country'] = np.where(df['native_country'] == ' United-States', 'United-States', 'Others')
df['workclass'] = np.where(df['workclass'] == ' ?', np.nan, df['workclass'])
df['occupation'] = np.where(df['occupation'] == ' ?', np.nan, df['occupation'])
df.dropna(inplace=True)
edu_label = {value: key for key, value in enumerate(df.education.unique())}
df['education'] = df['education'].map(edu_label)

wc_label = {value: key for key, value in enumerate(df.workclass.unique())}
df['workclass'] = df['workclass'].map(wc_label)

ms_label = {value: key for key, value in enumerate(df.marital_status.unique())}
df.marital_status = df.marital_status.map(ms_label)

occ_label = {value: key for key, value in enumerate(df.occupation.unique())}
df.occupation = df.occupation.map(occ_label)

r_label = {value: key for key, value in enumerate(df.race.unique())}
df.race =  df.race.map(r_label)

df['native_country'] = np.where(df['native_country'] == 'United-States', 1, 0)
rel_label = {value: key for key, value in enumerate(df.relationship.unique())}
df.relationship = df.relationship.map(rel_label)
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


lgbm.fit(X_train, y_train)
pred = lgbm.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
'''
Accuracy Score:  1.0
'''