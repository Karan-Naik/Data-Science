import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



svc=SVC()
RC=RandomForestClassifier(random_state=5)

scaler=StandardScaler()


df_train=pd.read_csv('C:/Users/manas/Downloads/train.csv/train.csv')
#print(df_train)

'''print(df_train.isnull().sum())
tBodyAcc-mean()-X       0
tBodyAcc-mean()-Y       0
tBodyAcc-mean()-Z       0
tBodyAcc-std()-X        0
tBodyAcc-std()-Y        0
                       ..
angle(X,gravityMean)    0
angle(Y,gravityMean)    0
angle(Z,gravityMean)    0
subject                 0
Activity                0
Length: 563, dtype: int64'''

X=pd.DataFrame(df_train.drop(['Activity','subject'],axis=1))
y=df_train.Activity.values.astype(object)


encoder=preprocessing.LabelEncoder()
encoder.fit(y)
y=encoder.transform(y)
#print(y.shape)

#print(y[5])
print(encoder.classes_)

X=scaler.fit_transform(X)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=100)

RC.fit(X_train,y_train)
y_pred=RC.predict(X_test)
print(accuracy_score(y_test,y_pred))
'''
Acc=0.981645139360979
'''
