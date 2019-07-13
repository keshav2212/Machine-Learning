# -*- coding: utf-8 -*-

from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
df=pd.read_csv("homeprices.csv")
median_bedroom=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(median_bedroom)
rg=linear_model.LinearRegression()
rg.fit(df[['area','bedrooms','age']],df.price)
print(t)
#plt.xlabel('area')
#plt.ylabel('price')
#plt.scatter(df.area,df.price,color="red",marker="*")
#plt.plot(df.area,df.price,color="blue")
#reg=linear_model.LinearRegression()
#reg.fit(df[['area']],df.price)
#t=reg.predict([[3300]])
#print(t)