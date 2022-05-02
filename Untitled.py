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



#df_ge = pd.read_csv('death_count.csv')
#df_ge.tail()

# plt.figure()
# plt.plot(df_ge["CIN"])
# plt.plot(df_ge["CFN"])
# plt.plot(df_ge["C"])
# plt.plot(df_ge["D"])
# plt.title('Corona')
# plt.ylabel('Count')
# plt.xlabel('Days')
# plt.legend(['CIN','CFN','C','D'], loc='upper left')
# plt.show()


#Setting start and end dates and fetching the historical data
 
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

#Data Preprocessing
stk_data['Date'] = stk_data.index
data2 = pd.DataFrame(columns = ['Date', 'CFN', 'CIN', 'C', 'D'])
data2['Date'] = stk_data['Date']
data2['CFN'] = stk_data['CFN']
data2['CIN'] = stk_data['CIN']
data2['C'] = stk_data['C']
data2['D'] = stk_data['D']
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


#Defining the LSTM Recurrent Model
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

#Compiling and fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)


#Fetching the test data and preprocessing
testdataframe = pd.read_csv('dataset.csv')

testdataframe['Date'] = testdataframe.index
testdata = pd.DataFrame(columns = ['Date', 'CFN', 'CIN', 'C', 'D'])
testdata['Date'] = testdataframe['Date']
testdata['CFN'] = testdataframe['CFN']
testdata['CIN'] = testdataframe['CIN']
testdata['C'] = testdataframe['C']
testdata['D'] = testdataframe['D']
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

#Visualizing the prediction
plt.figure(figsize=(20,10))
plt.plot(real_stock_price, color = 'green', label = 'SBI Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted SBI Stock Price')
plt.title('SBI Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('SBI Stock Price')
plt.legend()
plt.show()