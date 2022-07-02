import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

#GIVING COLNAMES AS DATASET DONT CONTAIN COL NAMES
colname=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

df=pd.read_csv("E:/Languages/Python/Practice/housing.csv",  header=None , delimiter=r"\s+" , names=colname)


LR=LinearRegression()

'''
#to get col names
print(df.columns.values)

#to get dimensions
print(np.shape(df))

print(df.describe())
'''

#drop values
x=df.drop(["MEDV","CHAS","RAD","B"],axis=1)
#y=df["MEDV"]
g = pd.cut(df["MEDV"],3,labels=['0','1','2'])
le = LabelEncoder()
le.fit(g)
y = le.transform(g)


'''
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
'''


#toconvert all data into one type
#X=np.asarray(X1,dtype=np.float64) #all values are float64
#Y=np.asarray(Y1,dtype=np.float64) #all values are float64


# to identify score
'''
bestfeatures = SelectKBest(score_func=chi2,k="all")
fit = bestfeatures.fit(x,y)
print(fit.scores_)
dfscores =pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores = pd.concat([dfcolumns, dfscores],axis=1)
featuresScores.columns=['Specs','Score']
print(featuresScores)
'''
'''
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(4).plot(kind="barh")
plt.show()
'''

'''
#categorical to numeric
le= LabelEncoder()
le.fit(y)
y=le.transform(y)
'''


'''
from matplotlib import pyplot as plt
import seaborn as sns
#sns.boxplot(x['CRIM'])
#plt.show()
#sns.boxplot(x['ZN'])
#plt.show()
sns.boxplot(x['INDUS'])
plt.show()
'''

'''
upperCRIM=25.0
upperZN=30
outCrim=x[x['CRIM']>upperCRIM].values
outZN=x[x['ZN']>upperZN].values

x['CRIM'].replace(outCrim,upperCRIM,inplace=True)
x['ZN'].replace(outZN,upperZN,inplace=True)

#sns.boxplot(x['CRIM'])
#plt.show()
'''

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

LR.fit(x_train,y_train)

y_pred=LR.predict(x_test)


acc=mean_squared_error(y_test,y_pred)
#print(y_pred)
print(acc)
'''
Acc=0.2039957000460529
'''