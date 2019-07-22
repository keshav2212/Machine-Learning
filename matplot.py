import pandas as pd
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6,7,8]
company=['google','apple','facebook','amazon','youtube','microsoft']
y=[83,63,53,86,67,25]
xpos=np.arange(len(company))
print(xpos)
#plt.plot(x,y,'g*--',label='keshav')
#plt.legend()
plt.xticks(xpos,company)
plt.bar(xpos,y)
t=df.groupby(df.Trend)
xpose=[]ac
for group in t:
    xpose=np.arange(len(group))
    ypose=group.mean()
print(ypose)
plt.bar(xpose,t['Total Traded Quantity'].mean())