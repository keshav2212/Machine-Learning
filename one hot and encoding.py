import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#df=pd.read_csv('one hot encoding.csv')
#df1=pd.get_dummies(df.town)
#df2=pd.concat([df,df1],axis='columns')
#df3=df2.drop(['town','west windsor'],axis='columns')
#rg=linear_model.LinearRegression()
#X=df3.drop(['price'],axis='columns')
#y=df3.price
#rg.fit(X,y)
#t=rg.predict([[2800,0,1]])
#plt.scatter(df['area'],df['price'])
#plt.plot(df['area'],df['price'])
#print(t)
df=pd.read_csv('insurance_data.csv')
X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)
rg=LogisticRegression()
rg.fit(X_train,y_train)
t=rg.predict(X_test)
print(rg.score(X_test,y_test))
print(t)
print(y_test)
