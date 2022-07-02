import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LR=LogisticRegression()
cv = CountVectorizer()


df_test=pd.read_json("E:/pythonProject/test.json")
df_train=pd.read_json("E:/pythonProject/train.json")

#print(df_train.head())


counters = {}
for cuisine in df_train['cuisine'].unique():
    counters[cuisine] = Counter()
    indices = (df_train['cuisine'] == cuisine)
    for ingredients in df_train[indices]['ingredients']:
        counters[cuisine].update(ingredients)

#print(counters['italian'].most_common(10))

top10 = pd.DataFrame([[items[0] for items in counters[cuisine].most_common(10)] for cuisine in counters],
            index=[cuisine for cuisine in counters],
            columns=['top{}'.format(i) for i in range(1, 11)])
#print(top10)
df_train['all_ingredients'] = df_train['ingredients'].map(";".join)
#print(df_train.head())

X = cv.fit_transform(df_train['all_ingredients'].values)
#print(list(cv.vocabulary_.keys())[:100])


enc = LabelEncoder()
y = enc.fit_transform(df_train.cuisine)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

LR.fit(X_train,y_train)

y_pred=LR.predict(X_test)

print(accuracy_score(y_test,y_pred))

'''
Acc=0.7903205531112508
'''