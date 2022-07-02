import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


LR=LogisticRegression(random_state=0)
RF=RandomForestClassifier(random_state=1)
GB=GradientBoostingClassifier(n_estimators=10)
SM=svm.SVC()
MLP=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
RFR=RandomForestRegressor(n_jobs=-1)
df = pd.read_csv('C:/Users/manas/Downloads/Black Friday Sales/train.csv')
x_test= pd.read_csv('C:/Users/manas/Downloads/Black Friday Sales/test.csv')
#print(df.head())

#print(df.describe())
'''
            User_ID     Occupation  ...  Product_Category_3       Purchase
count  5.500680e+05  550068.000000  ...       166821.000000  550068.000000
mean   1.003029e+06       8.076707  ...           12.668243    9263.968713
std    1.727592e+03       6.522660  ...            4.125338    5023.065394
min    1.000001e+06       0.000000  ...            3.000000      12.000000
25%    1.001516e+06       2.000000  ...            9.000000    5823.000000
50%    1.003077e+06       7.000000  ...           14.000000    8047.000000
75%    1.004478e+06      14.000000  ...           16.000000   12054.000000
max    1.006040e+06      20.000000  ...           18.000000   23961.000000
'''

#print(df.info())
'''

 #   Column                      Non-Null Count   Dtype  
---  ------                      --------------   -----  
 0   User_ID                     550068 non-null  int64  
 1   Product_ID                  550068 non-null  object 
 2   Gender                      550068 non-null  object 
 3   Age                         550068 non-null  object 
 4   Occupation                  550068 non-null  int64  
 5   City_Category               550068 non-null  object 
 6   Stay_In_Current_City_Years  550068 non-null  object 
 7   Marital_Status              550068 non-null  int64  
 8   Product_Category_1          550068 non-null  int64  
 9   Product_Category_2          376430 non-null  float64
 10  Product_Category_3          166821 non-null  float64
 11  Purchase                    550068 non-null  int64 '''

df['Product_Category_2'] = df['Product_Category_2'].fillna(-2.0).astype("float32")
df['Product_Category_3'] = df['Product_Category_3'].fillna(-2.0).astype("float32")
#print(df.info())

#print(df.apply(lambda x: len(x.unique())))

gender_dict = {'F':0, 'M':1}
df['Gender'] = df['Gender'].apply(lambda x: gender_dict[x])


cols = ['Age', 'City_Category', 'Stay_In_Current_City_Years']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])
#print(df.head())

X = df.drop(columns=['User_ID', 'Product_ID', 'Purchase'])
y = df['Purchase']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
'''
#LR.fit(x_train,y_train)
RF.fit(x_train,y_train)
GB.fit(x_train,y_train)
SM.fit(x_train,y_train)
MLP.fit(x_train,y_train)

#predLR=LR.predict(x_test)
predRF=RF.predict(x_test)
predGB=GB.predict(x_test)
predSM=SM.predict(x_test)
predMLP=MLP.predict(x_test)

#print(accuracy_score(y_test,predLR))
print(accuracy_score(y_test,predRF))
print(accuracy_score(y_test,predGB))
print(accuracy_score(y_test,predSM))
'''
RFR.fit(x_train,y_train)
y_pred=RFR.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
'''
3044.7037003022324
'''
submission = pd.DataFrame()
submission['User_ID'] = x_test['User_ID']
submission['Purchase'] = y_pred
submission.to_csv('submission.csv',index=False)
