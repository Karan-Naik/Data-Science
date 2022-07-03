import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

RFR = RandomForestRegressor(n_estimators=100,n_jobs=-1)
LR=LogisticRegression(random_state=0)
RF=RandomForestClassifier(random_state=1)
GB=GradientBoostingClassifier(n_estimators=10)
DT=DecisionTreeClassifier(random_state=0)
SM=svm.SVC()
MLP=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
MB=MultinomialNB()
GN=GaussianNB()



data=pd.read_csv("C:/Users/manas/Downloads/walmart-recruiting-store-sales-forecasting/train.csv")
data_test=pd.read_csv("C:/Users/manas/Downloads/walmart-recruiting-store-sales-forecasting/test.csv")

print(data.head())

'''
print(data.info())
Data columns (total 5 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   Store         421570 non-null  int64  
 1   Dept          421570 non-null  int64  
 2   Date          421570 non-null  object 
 3   Weekly_Sales  421570 non-null  float64
 4   IsHoliday     421570 non-null  bool   

'''
'''
gender_dict = {'False':0, 'True':1}
data['IsHoliday'] = data['IsHoliday'].apply(lambda x: gender_dict[x])

print(data.head())

'''

X_t=data.drop(['Weekly_Sales','IsHoliday','Date'],axis=1)
Y_t=data['Weekly_Sales']

print(X_t)
print(Y_t)

x_train,x_test,y_train,y_test=train_test_split(X_t,Y_t,random_state=0,test_size=0.2)



'''
#LR.fit(x_train,y_train)
RF.fit(x_train,y_train)
GB.fit(x_train,y_train)
DT.fit(x_train,y_train)
SM.fit(x_train,y_train)
MLP.fit(x_train,y_train)
MB.fit(x_train,y_train)
GN.fit(x_train,y_train)

#to test the modle and store results in var
#y_pred=LR.predict(x_test)
y_pred1=RF.predict(x_test)
y_pred2=GB.predict(x_test)
y_pred3=DT.predict((x_test))
y_pred4=SM.predict(x_test)
y_pred5=MLP.predict(x_test)
y_pred6=MB.predict(x_test)
y_pred7=GN.predict(x_test)

#print("Logistic",accuracy_score(y_test,y_pred))
print("Random Forest",accuracy_score(y_test,y_pred1))
print("Gradient Boosting",accuracy_score(y_test,y_pred2))
print("Decision Tress",accuracy_score(y_test,y_pred3))
print("Svm",accuracy_score(y_test,y_pred4))
print("MLB Class",accuracy_score(y_test,y_pred5))
print("Multinomial",accuracy_score(y_test,y_pred6))
print("gaussian",accuracy_score(y_test,y_pred7))'''

RFR.fit(x_train,y_train)

y_pred=RFR.predict(x_test)
print(y_pred)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
'''
6904.3324170458
'''


