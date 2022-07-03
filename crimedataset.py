import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


RFR=RandomForestRegressor(n_jobs=-1)

df1 = pd.read_csv('C:/Users/manas/Downloads/Chicago_Crimes_2001_to_2004.csv/Chicago_Crimes_2001_to_2004.csv',low_memory=False ,on_bad_lines='skip')
#df2 = pd.read_csv('C:/Users/manas/Downloads/Chicago_Crimes_2005_to_2007.csv/Chicago_Crimes_2005_to_2007.csv',low_memory=False, on_bad_lines='skip')
#df3 = pd.read_csv('C:/Users/manas/Downloads/Chicago_Crimes_2008_to_2011.csv/Chicago_Crimes_2008_to_2011.csv',low_memory=False, on_bad_lines='skip')
#df4 = pd.read_csv('C:/Users/manas/Downloads/Chicago_Crimes_2012_to_2017.csv/Chicago_Crimes_2012_to_2017.csv',low_memory=False, on_bad_lines='skip')
#df = pd.concat([df1, df2, df3,df4], ignore_index=False, axis=0)

#print(df1.keys())

df1=df1.dropna()
x=df1.drop(['Unnamed: 0','Beat','Ward','Year', 'ID', 'Case Number','Date' ,'Block','IUCR','Primary Type','Description','Location Description','Arrest','Domestic', 'FBI Code',  'Y Coordinate',  'Updated On' ,'Latitude', 'Location'] ,axis=1)
y=df1['Location']


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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

RFR.fit(x_train,y_train)

y_pred=RFR.predict(x_test)

print(mean_squared_error(y_test,y_pred))
