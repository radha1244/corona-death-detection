#!/usr/bin/env python
# coding: utf-8

# In[111]:


pip install pycountry


# In[113]:


pip install pywaffle


# In[114]:


pip install folium


# In[185]:


# Import Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly
import seaborn as sns
sns.set()
import pycountry
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go

import plotly.offline as py

from pywaffle import Waffle

py.init_notebook_mode(connected=True)
import folium 
from folium import plugins
plt.style.use("fivethirtyeight")# for pretty graphs

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[186]:


# Importing dataset   
data3 = pd.read_csv('covid_19_india.csv')
data1=data3.copy()


# In[187]:


data1.head()
data1.drop(["Sno", "Time"], axis=1, inplace = True)
s2=data1['Date']
# Create computed Col <Total cases> 
data1['Total_cases'] = data1['ConfirmedIndianNational'] + data1['ConfirmedForeignNational']

# Create computed col <Active Cases>
data1['Active_cases'] = data1['Total_cases'] - (data1['Cured'] + data1['Deaths'])


# In[188]:



#data['Date'] = data['Date'].astype('datetime64[D]')
data1.head()


# In[17]:


X = data.iloc[:,0:4].values

y = data.iloc[:,4].values


# In[189]:


fig = px.bar(data, x='Date', y='Total_cases',hover_data=['Deaths'], color='Deaths',height=400)
fig.show()


# In[183]:


# Plot EU countries spread 
cols_keep = ['Date','Total_cases','Cured','Deaths']

# subset df
data1= data1[cols_keep]


# get Date DF
DT_df = data1[['Date']]
DT_df  = DT_df.set_index('Date')
print(DT_df)

# Set Index to Date 
data1 = data1.set_index('Date')

data1.head()
data1.plot()


# In[139]:



#--------------------------------------
# Model US 
#--------------------------------------
# Y data 
#s2=data['Date']
s1=data1['Total_cases']
Y=s1
print(Y)
#Y.shape()
# X data 
X = np.arange(1,len(Y)+1)
Xdate = s2
# Fit 3rd Degree polynomial capture coefficients 
Z = np.polyfit(X,Y,1)
# Generate polynomial function with these coefficients 
P = np.poly1d(Z)
# Generate X data for forecast 
XP = np.arange(1,len(Y)+8)
# Generate forecast 
YP = P(XP)
# Fit Curve
Yfit = P(X)

import datetime
start = Xdate[0]
end_dt = datetime.datetime.strptime(Xdate[len(Xdate)-1], "%Y-%m-%d")
end_date = datetime.datetime.strptime(str(end_dt),'%Y-%m-%d %H:%M:%S').date()
end_forecast_dt= end_dt + datetime.timedelta(days=7)
end_forecast =  datetime.datetime.strptime(str(end_forecast_dt),'%Y-%m-%d %H:%M:%S').date()
#
mydates = pd.date_range(start, end_forecast).to_list()
mydates_df = pd.DataFrame(mydates,columns =['Date']) 
mydates_df  = mydates_df.set_index('Date')
mydates_df['Date'] = mydates_df.index
X_FC = mydates_df['Date']
print(X)
print(Y)


# In[140]:


fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)
ax.plot(X, Y, label='Actual Confirmed')
ax.plot(XP, YP, '--',label='Predicted Fit using 3rd degree polynomial')
plt.title('COVID RISE IN India Current Vs Preditions till 1st April 2020')
ax.legend()
plt.show()


# In[141]:




fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111)
ax.plot(Xdate, Y, label='Actual Confirmed')
ax.plot(Xdate, Yfit, '--',label='Predicted Fit using 3rd degree polynomial')
plt.title('COVID RISE in US : 3rd Degree polynomial Fit')
ax.legend()
plt.show()


# In[142]:


fig, ax = plt.subplots(figsize=(20,10))
ax.plot(Xdate,Y,'b-')
ax.tick_params(direction='out', length=10, width=10, colors='r')
ax.set_xlabel('Date',fontsize=25)
ax.set_ylabel('Confirmed Cases',fontsize=25)
ax.set_title('COVID 19 Spread in US as of 24th March 2020',fontsize=25)
ax.set_ylim(0,180000)
fig.autofmt_xdate()

ax.grid(True)
fig.tight_layout()

plt.show()


# In[ ]:





# In[124]:


# Define new figure 
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(X_FC,YP,'--')
ax.tick_params(direction='out', length=10, width=10, colors='r')
ax.set_xlabel('Date',fontsize=25)
ax.set_ylabel('Predicted Cases',fontsize=25)
ax.set_ylim(0,180000)
ax.set_title('COVID 19 PREDICTION for US till 1st April 2020',fontsize=25)
fig.autofmt_xdate()

ax.grid(True)
fig.tight_layout()

plt.show()


# In[61]:



plt.figure(figsize=(25,10))
#fig,ax=plt.subplots(figsize=(12,10),edgecolor='k')
plt.plot(data)
#x.show()


# In[148]:


Total_cases  = data1['Total_cases'].sum()
Total_active = data1['Active_cases'].sum()
Total_deaths = data1['Deaths'].sum()
Total_cured  = data1['Cured'].sum()


# Create Dict 
dict1 = {'Total cases':Total_cases, 'Active cases':Total_active,'Total deaths':Total_deaths,'Total cured':Total_cured}

# Convert to DF 
df_totals= pd.DataFrame.from_dict(dict1,orient='index',columns=['Count'])
df_totals = df_totals.reset_index()

df_totals.rename(columns= {'index':'Case Classification'},inplace= True)


# In[149]:


fig = px.bar(df_totals, x='Case Classification', y='Count',
             hover_data=['Count'], color='Count',
             labels={'Count':'Count'},text = 'Count',height=400)
fig.show()


# In[151]:


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: pink' if v else '' for v in is_max]


data1.style.apply(highlight_max,subset=['Recovered', 'Deaths','Total_cases','Active_cases'])


# In[158]:


fig = px.bar(data1.sort_values('Active_cases', ascending=True),x="Active_cases", y="State/UnionTerritory", title='Total Active Cases', text='Active_cases', orientation='h',width=700, height=700, range_x = [0, max(data1['Active_cases'])],color = 'Active_cases',color_continuous_scale=px.colors.sequential.thermal)


fig.show()


# In[181]:


# Plotting with subplots and Seaborne 

Total_cases  = data1['Total_cases'].sum()
Total_active = data1['Active_cases'].sum()
Total_deaths = data1['Deaths'].sum()
Total_cured  = data1['Cured'].sum()

# define subplot instance 
f, ax = plt.subplots(figsize=(20,12))

# Subset only relevant data 
data = data1[['State/UnionTerritory','Total_cases','Cured','Deaths']]

# Sort descending on Total Cases 
data.sort_values('Total_cases',ascending=False,inplace=True)

# Set Seaborne parameters
sns.set_color_codes("pastel")


sns.barplot(x="Total_cases", y="State/UnionTerritory", data=data,label="Total_cases", color="r")

sns.set_color_codes("muted")

sns.barplot(x="Cured", y="State/UnionTerritory", data=data,label="Recovered", color="g")


# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 35), ylabel="",xlabel="Cases")
sns.despine(left=True, bottom=True)


# In[ ]:


fig,ax=plt.subplots(figsize=(12,10),edgecolor='k')

plt.bar(data['State/UnionTerritory'], data['ConfirmedForeignNational'],w,color='b') 

plt.xlabel("State/UnionTerritory") 
plt.ylabel("ConfirmedForeignNational") 

plt.show() 


# In[ ]:


#data.plot.bar() 
fig,ax=plt.subplots(figsize=(12,10),edgecolor='k')

# plot between 2 attributes 
plt.bar(data['State/UnionTerritory'], data['ConfirmedIndianNational'],w,color='b') 
plt.xlabel("State/UnionTerritory") 
plt.ylabel("ConfirmedIndianNational") 

plt.show() 


# In[ ]:


w=0.20
fig,ax=plt.subplots(figsize=(12,10),edgecolor='k')
plt.bar(data['State/UnionTerritory'], data['Cured'],w,color='b')
plt.xlabel("State/UnionTerritory")
plt.ylabel("Cured") 


plt.show() 


# In[ ]:


#data.drop("Sno", "Time"], axis = 1, inplace = True)
X = data.iloc[:,0:3].values

y = data.iloc[:,3].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#import libray
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#for the value of slope and cooficient
print(regressor.intercept_)
#coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(X_test)
print(y_test)
#print(coeff_df)
y_pred = regressor.predict(X_test)
print(y_pred)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


from sklearn import tree

model=tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
#model.score(,target)
print(X_test)
print(model.predict([[165,8,11]]))
 


# In[ ]:


import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[ ]:


fig = px.bar(df_totals, x='Country', y='Num Days with > 100 Cases',
             hover_data=['Num Days with > 100 Cases'], color='Num Days with > 100 Cases',
             labels={'Num Days with > 100 Cases':'Num Days with > 100 Cases'},text = 'Num Days with > 100 Cases',height=400)
fig.show()


# In[ ]:


pip install tensorflow


# In[ ]:





# In[32]:


stk_data  = pd.read_csv('death_count.csv')

#Visualizing the fetched data
plt.figure()
plt.plot(stk_data["CIN"])
plt.plot(stk_data["CFN"])
plt.plot(stk_data["C"])
plt.plot(stk_data["D"])
plt.title('Corona')
plt.ylabel('Count')
plt.xlabel('Days')
plt.legend(['CIN','CFN','C','D'], loc='upper left')
#plt.show()


# In[ ]:


tensorflow


# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


import datetime as dt


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


from sklearn import model_selection


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:





# In[ ]:


pip install keras


# In[ ]:


pip install keras.model


# In[ ]:


from tensorflow.keras import layers


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import Sequential


# In[ ]:


import tensorflow as tf


# In[ ]:


print(tf.__version__)


# In[ ]:


pip show tensorflow


# In[ ]:


pip install --upgrade tensorflow


# In[ ]:


import tensorflow as tf


# In[ ]:


pip install --downgrade tensorflow


# In[ ]:


pip install tensorflow==1.1


# In[ ]:


pip install tensorflow==1.13.1


# In[ ]:


stk_data  = pd.read_csv('covid_19-india.csv')

#Visualizing the fetched data
plt.figure()
plt.plot(stk_data["CIN"])
plt.plot(stk_data["CFN"])
plt.plot(stk_data["C"])
plt.plot(stk_data["D"])
plt.title('Corona')
plt.ylabel('Count')
plt.xlabel('Days')
plt.legend(['CIN','CFN','C','D'], loc='upper left')
#plt.show()


# In[ ]:


stk_data = pd.read_csv('death_count.csv')


# In[ ]:


#Visualizing the fetched data
plt.figure()
plt.plot(stk_data["CIN"])
plt.plot(stk_data["CFN"])
plt.plot(stk_data["C"])
plt.plot(stk_data["D"])
plt.title('Corona')
plt.ylabel('Count')
plt.xlabel('Days')
plt.legend(['CIN','CFN','C','D'], loc='upper left')
#plt.show()


# In[ ]:


stk_data['Date'] = stk_data.index
data2 = pd.DataFrame(columns = ['Date', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths'])
data2['Date'] = stk_data['Date']
data2['CFN'] = stk_data['ConfirmedIndianNational']
data2['CIN'] = stk_data['ConfirmedForeignNational']
data2['C'] = stk_data['Cured']
data2['D'] = stk_data['Deaths']
train_set = data2.iloc[:, 1:5].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)
print(len(training_set_scaled))
X_train = []
y_train = []
for i in range(2, 43):
	X_train.append(training_set_scaled[i-2:i, 0])
	y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))


# In[ ]:


#Compiling and fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)


# In[ ]:


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[ ]:


testdataframe = pd.read_csv('death_count.csv')
testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths'])
testdata['Date'] = testdataframe['Date']
testdata['CFN'] = testdataframe['ConfirmedIndianNational']
testdata['CIN'] = testdataframe['ConfirmedForeignNational']
testdata['C'] = testdataframe['Cured']
testdata['D'] = testdataframe['Deaths']
real_stock_price = testdata.iloc[:, 1:2].values
dataset_total = pd.concat((data2['CFN'], testdata['CFN']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(testdata) - 0:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(2, 43):
	X_test.append(inputs[i-2:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Making predictions on the test data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[ ]:


testdata1 = pd.read_csv('IndividualDetails.csv')


# In[ ]:


data.head()


# In[ ]:




