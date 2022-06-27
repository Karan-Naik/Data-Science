import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

#creating instance of algo
LR=LogisticRegression(random_state=0)
RF=RandomForestClassifier(random_state=1)
GB=GradientBoostingClassifier(n_estimators=10)
DT=DecisionTreeClassifier(random_state=0)
SM=svm.SVC()
MLP=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
MB=MultinomialNB()
GN=GaussianNB()


df=pd.read_csv("D:/Karan Naik/IRIS.csv")

#to identify x and y
X=df.drop("species",axis=1) #axis =1 delete the holl colum
Y=df["species"] # selects the specified

#to divide datset into 70-30 , random state shufle on bases of integer <format is same>
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.3)

#to train the modle
LR.fit(x_train,y_train)
RF.fit(x_train,y_train)
GB.fit(x_train,y_train)
DT.fit(x_train,y_train)
SM.fit(x_train,y_train)
MLP.fit(x_train,y_train)
MB.fit(x_train,y_train)
GN.fit(x_train,y_train)

#to test the modle and store results in var
y_pred=LR.predict(x_test)
y_pred1=RF.predict(x_test)
y_pred2=GB.predict(x_test)
y_pred3=DT.predict((x_test))
y_pred4=SM.predict(x_test)
y_pred5=MLP.predict(x_test)
y_pred6=MB.predict(x_test)
y_pred7=GN.predict(x_test)

#to check accuracy
print("Logistic",accuracy_score(y_test,y_pred))
print("Random Forest",accuracy_score(y_test,y_pred1))
print("Gradient Boosting",accuracy_score(y_test,y_pred2))
print("Decision Tress",accuracy_score(y_test,y_pred3))
print("Svm",accuracy_score(y_test,y_pred4))
print("MLB Class",accuracy_score(y_test,y_pred5))
print("Multinomial",accuracy_score(y_test,y_pred6))
print("gaussian",accuracy_score(y_test,y_pred7))

# accuracy score shall never be 100%

'''Accuracy
Logistic 0.9777777777777777
Random Forest 0.9777777777777777
Gradient Boosting 0.9777777777777777
Decision Tress 0.9777777777777777
Svm 0.9777777777777777
MLB Class 0.24444444444444444
Multinomial 0.6
gaussian 1.0'''


