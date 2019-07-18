import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib
import pickle
import math
#joblib and pickle are same you can use any one of them 
#we use joblib when we have large number of numpy array
#only write :- joblib.dump(model_name,file_name)
#for loading write load instead of dump
df=pd.read_csv('homeprices.csv')
t=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(t)
rg=linear_model.LinearRegression()
rg.fit(df[['area']],df.price)
with open('model_pickle','wb') as f:
    pickle.dump(rg,f)
with open('model_pickle','rb') as f:
    tm=pickle.load(f)
g=tm.predict([[35000]])
print(g)
