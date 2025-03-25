"""
File: Atl3_MLP_build.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/Yaldron
Description: 
"""
import numpy  as np
import xarray as  xr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


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
# 定义模型
model = Sequential([
    Flatten(input_shape=(108,)),  # 9X12
    Dense(64, activation='tanh'), #use_bias=True,
    Dropout(0.1),
    Dense(128, activation='tanh'),
    Dropout(0.1),
    Dense(64, activation='tanh'),
    Dropout(0.1),
    Dense(1, activation='tanh')
])


# 编译模型
model.compile(optimizer='adam',
              loss='mse')

# 屏幕输出模型结构
model.summary()
#=================================================================

# 组织训练数据 和 验证数据

# year 1855~1979

for ist in [3]: #range(12):
	X_train = np.zeros((1979-1855+1,108),dtype=np.float32)
	X_valid = np.zeros((2023-1980+1,108),dtype=np.float32)

	for iyr in range(1855,1979):
		X_train[iyr-1855,0:12]   = Nino3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,12:24]  = Nino4[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,24:36]  = Nino34[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,36:48]  = Nino12[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,48:60]  = EIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,60:72]  = WIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,72:84]  = TNA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,84:96]  = TSA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_train[iyr-1855,96:108] = Atl3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]

	for iyr in range(1980,2023):
		X_valid[iyr-1980,0:12]  = Nino3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,12:24] = Nino4[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,24:36] = Nino34[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,36:48] = Nino12[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,48:60] = EIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,60:72] = WIOD[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,72:84] = TNA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,84:96] = TSA[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]
		X_valid[iyr-1980,96:108] = Atl3[(iyr-1854)*12-12+ist:(iyr-1854)*12+ist]

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
plt.savefig('Prediction_Atl3_MLP_st-'+str(ist+1).zfill(2)+'_lead-'+str(ilead+1).zfill(2)+'.svg',dpi=800,bbox_inches = 'tight')

model.save('Atl3_MLP_st-'+str(ist+1).zfill(2)+'_lead-'+str(ilead+1).zfill(2)+'.h5')




