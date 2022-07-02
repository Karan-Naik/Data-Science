import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


LR=LinearRegression()
MB=MultinomialNB()
GB=GaussianNB()
RF=RandomForestClassifier(random_state=1)
GBC=GradientBoostingClassifier(n_estimators=10)


df_red = pd.read_csv("E:/pythonProject/winequality_red.csv")
df_white = pd.read_csv("E:/pythonProject/winequality_white.csv")

df_red['color'] = 'Red'
df_white['color'] = 'White'
df = pd.concat([df_red,df_white], axis=0)

X = df.drop('color', axis=1)
X = X.drop('quality', axis=1)
y = df['quality']

miss=pd.DataFrame(np.round(df.isna().sum()/len(df)*100,2), columns=['percentage_missing'])
print(miss)

bestfeatures = SelectKBest(score_func=chi2,k="all")
fit = bestfeatures.fit(X,y)
print(fit.scores_)
dfscores =pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores],axis=1)
featuresScores.columns=['Specs','Score']
print(featuresScores)

'''
                   Specs        Score
0          fixed acidity    11.115118
1       volatile acidity    42.528980
2            citric acid     3.673977
3         residual sugar   225.657151
4              chlorides     6.388650
5    free sulfur dioxide   913.332832
6   total sulfur dioxide  1269.974108
7                density     0.006600
8                     pH     0.097332
9              sulphates     1.078473
10               alcohol   201.369195
'''
X=X.drop('density',axis=1)
X=X.drop('pH',axis=1)
X=X.drop('sulphates',axis=1)
X=X.drop('chlorides',axis=1)
X=X.drop('citric acid',axis=1)

'''from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(X['fixed acidity'])
plt.show()
'''


print(X.keys())

X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=0,test_size=0.2)

LR.fit(X_train,y_train)
MB.fit(X_train,y_train)
GB.fit(X_train,y_train)
RF.fit(X_train,y_train)
GBC.fit(X_train,y_train)


y_pred=LR.predict(X_test)
y_pred1=MB.predict(X_test)
y_pred2=GB.predict(X_test)
y_pred3=RF.predict(X_test)
y_pred4=GBC.predict(X_test)


acc=mean_squared_error(y_test,y_pred)
acc1=accuracy_score(y_test,y_pred1)
acc2=accuracy_score(y_test,y_pred2)
acc3=accuracy_score(y_test,y_pred3)
acc4=accuracy_score(y_test,y_pred4)

print('Linear',acc)
print("MB",acc1)
print("GB",acc2)
print("RF",acc3)
print("GBC",acc4)


'''
Linear 0.5498670652930597
MB 0.36230769230769233
GB 0.46923076923076923
RF 0.6784615384615384
GBC 0.5330769230769231
'''
