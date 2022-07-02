import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

LOR=LogisticRegression(random_state=0)
LR=LinearRegression()


df_test=pd.read_csv("E:/pythonProject/test.csv")
df_train=pd.read_csv("E:/pythonProject/train.csv")
#print(df_test.keys())
#print(df_train.keys())

X_train=df_train.drop(['Survived','PassengerId','Age','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
Y_train=df_train['Survived']

#print(X_train)
#print(Y_train)


bestfeatures = SelectKBest(score_func=chi2,k="all")
fit = bestfeatures.fit(X_train,Y_train)
#print(fit.scores_)
dfscores =pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X_train.columns)
featuresScores = pd.concat([dfcolumns, dfscores],axis=1)
featuresScores.columns=['Specs','Score']
#print(featuresScores)

'''
    Specs        Score
0  Pclass    30.873699
1   SibSp     2.581865
2   Parch    10.097499
3    Fare  4518.319091
'''

X_train=X_train.drop('SibSp',axis=1)
miss=pd.DataFrame(np.round(X_train.isna().sum()/len(X_train)*100,2), columns=['percentage_missing'])
#print(miss)
'''
        percentage_missing
Pclass                 0.0
Parch                  0.0
Fare                   0.0
'''
#print(X_train)

X_test=df_test.drop(['SibSp','PassengerId','Age','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
miss1=pd.DataFrame(np.round(X_test.isna().sum()/len(X_test)*100,2), columns=['percentage_missing'])
#print(miss1)
'''
        percentage_missing
Pclass                0.00
Parch                 0.00
Fare                  0.24
'''
#print(X_test.describe())
'''
count  418.000000  418.000000  417.000000
mean     2.265550    0.392344   35.627188
std      0.841838    0.981429   55.907576
min      1.000000    0.000000    0.000000
25%      1.000000    0.000000    7.895800
50%      3.000000    0.000000   14.454200
75%      3.000000    0.000000   31.500000
max      3.000000    9.000000  512.329200

'''

fare_mean=35.627188
X_test['Fare']=X_test['Fare'].fillna(fare_mean)

miss2=pd.DataFrame(np.round(X_test.isna().sum()/len(X_test)*100,2), columns=['percentage_missing'])
'''print(miss2)
        percentage_missing
Pclass                 0.0
Parch                  0.0
Fare                   0.0
'''


LOR.fit(X_train,Y_train)

acc=LOR.predict(X_test)
submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': acc})
submission.to_csv('submission.csv',index=False)
