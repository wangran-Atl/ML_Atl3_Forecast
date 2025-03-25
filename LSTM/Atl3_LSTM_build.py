"""
File: Atl3_LSTM_build.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/Yaldron
Description: 
"""
import os
import numpy  as np
import xarray as  xr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM


f = xr.open_dataset('../post_data/Climate_IDX_Ocn_1854-2024.nc')

Nino3  = f['Nino3']
Nino4  =  f['Nino4']
Nino34 =  f['Nino34']
Nino12 =  f['Nino12']
WIOD   =  f['WIOD']
EIOD   =  f['EIOD']
Atl3   =  f['Atl3']
TNA    =  f['TNA']
TSA    =  f['TSA']


#=================================================================

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, activation='tanh', input_shape=(12, 9)))
model.add(LSTM(units=50, return_sequences=True,activation='tanh'))
model.add(LSTM(units=30, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer=keras.optimizers.RMSprop(0.005,0.9), loss='mse')
model.summary()

#=================================================================

# 组织训练数据 和 验证数据

# year 1855~1979

for ist in [3]: #range(12):
	X_train = np.zeros((1979-1855+1,12,9),dtype=np.float32)
	X_valid = np.zeros((2023-1980+1,12,9),dtype=np.float32)

	for iyr in range(1855,1979):
		X_train[iyr-1855,:,0]   = Nino3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,1]   = Nino4[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,2]   = Nino34[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,3]   = Nino12[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,4]   = EIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,5]   = WIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,6]   = TNA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,7]   = TSA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,:,8]   = Atl3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]

	for iyr in range(1980,2023):
		X_valid[iyr-1980,:,0]  = Nino3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,1]  = Nino4[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,2]  = Nino34[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,3]  = Nino12[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,4]  = EIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,5]  = WIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,6]  = TNA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,7]  = TSA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,:,8]  = Atl3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]

	for ilead in [2]: #range(12):
		Y_train = np.zeros((1979-1855+1,1),dtype=np.float32)
		Y_valid = np.zeros((2023-1980+1,1),dtype=np.float32)

		for iyr in range(1855,1979):
			Y_train[iyr-1855,0] = Atl3[(iyr-1854)*12+ist+ilead]
		for iyr in range(1980,2023):
			Y_valid[iyr-1980,0] = Atl3[(iyr-1854)*12+ist+ilead]


		model.fit(X_train,Y_train,epochs=30)

		loss = model.evaluate(X_valid,Y_valid)

		Y_pred = model.predict(X_valid)

		print('RMSE for MLP start '+str(ist+1).zfill(2)+' lead '+str(ilead+1).zfill(2)+'is ',loss)




import matplotlib.pyplot as plt


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5)) #width+height

ax.plot(Y_valid,'ko--')
ax.plot(Y_pred,'ro--')

print('ACC is ',np.corrcoef(Y_valid.reshape(-1),Y_pred.reshape(-1))[0][1])
plt.savefig('Prediction_Atl3_LSTM_st-'+str(ist+1).zfill(2)+'_lead-'+str(ilead+1).zfill(2)+'.svg',dpi=800,bbox_inches = 'tight')

model.save('Atl3_LSTM_st-'+str(ist+1).zfill(2)+'_lead-'+str(ilead+1).zfill(2)+'.h5')




