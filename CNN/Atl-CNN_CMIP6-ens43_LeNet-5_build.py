"""
File: CNN_sst+zos.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/yourname
Description:
"""
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from   netCDF4 import Dataset

import scipy      as sp
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

##############################################################################
fsst1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_tos_historical_anom_detrend.nc"
fzos1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zos_historical_anom_detrend.nc"
fpsl1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zg_historical_anom_detrend_850hPa.nc"
fhgt1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zg_historical_anom_detrend_500hPa.nc"

fsst2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_ersstv5_198001-201812_Ham-LR.nc"
fzos2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_sshg_GODAS_198001-201812_Ham-LR.nc"
fpsl2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_hgt_JRA55_mon_198001-201812_850hPa_Ham-LR.nc"
fhgt2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_hgt_JRA55_mon_198001-201812_500hPa_Ham-LR.nc"

fSST1   = xr.open_dataset("/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_tos_historical_anom_detrend_HR.nc")
fSST2   = xr.open_dataset("/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_ersstv5_198001-201812_Ham-HR.nc")

Atl3_1 = fSST1['tos'].loc[:,:,-3:3,340:360].mean(("lat","lon")).values
Atl3_2 = fSST2['tos'].loc[:,-3:3,340:360].mean(("lat","lon")).values

#os._exit(0)

nc11 = Dataset(fsst1)
nc12 = Dataset(fzos1)
nc13 = Dataset(fpsl1)
nc14 = Dataset(fhgt1)

nc21 = Dataset(fsst2)
nc22 = Dataset(fzos2)
nc23 = Dataset(fpsl2)
nc24 = Dataset(fhgt2)


ssta1 = nc11.variables['tos'][:]
zosa1 = nc12.variables['zos'][:]
psla1 = nc13.variables['zg'][:,:,0,:,:]
hgta1 = nc14.variables['zg'][:,:,0,:,:]



ssta2 = nc21.variables['tos'][:]
zosa2 = nc22.variables['zos'][:]
psla2 = nc23.variables['hgt'][:,0,:,:]
hgta2 = nc24.variables['hgt'][:,0,:,:]



lat = nc11.variables['lat'][:]
lon = nc11.variables['lon'][:]

nlat = lat[:].shape[0]
nlon = lon[:].shape[0]


###############  arrange the X data and Y data  ###############################
monst = 3 #int(os.environ.get("NMON"))
nstep = 2 #int(os.environ.get("NSTEP"))

nstep = nstep - 1

print('nlat', nlat)
print('nlon', nlon)

nlen = 140
nens = 43
ntrain = nlen * nens
nvali = 36

X = np.zeros((ntrain, nlat, nlon, 12), dtype=float)
Y = np.zeros((ntrain, 1), dtype=float)  # (150*3)

XT = np.zeros((nvali, nlat, nlon, 12), dtype=float)
YT = np.zeros((nvali, 1), dtype=float)  # (5*3)

for i in range(3):
	for j in range(nens):
		X[nlen * j:nlen * (j + 1), :, :, i]     = ssta1[j, 12 + monst - 1 - i:12 + monst - 1 - i + nlen * 12:12, :, :]
		X[nlen * j:nlen * (j + 1), :, :, i + 3] = zosa1[j, 12 + monst - 1 - i:12 + monst - 1 - i + nlen * 12:12, :, :]*10.0
		X[nlen * j:nlen * (j + 1), :, :, i + 6] = psla1[j, 12 + monst - 1 - i:12 + monst - 1 - i + nlen * 12:12, :, :]/10.0
		X[nlen * j:nlen * (j + 1), :, :, i + 9] = hgta1[j, 12 + monst - 1 - i:12 + monst - 1 - i + nlen * 12:12, :, :]/10.0  
	XT[:, :, :, i]     = ssta2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]
	XT[:, :, :, i + 3] = zosa2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]*10.0
	XT[:, :, :, i + 6] = psla2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]/10.0
	XT[:, :, :, i + 9] = hgta2[12 + monst - 1 -i:12 + monst - 1 - i + nvali * 12:12, :, :]/10.0


print(psla1.shape)
for j in range(nens):
	tmp1 = Atl3_1[j, 12 + monst + nstep-1:12 +monst + nstep + nlen * 12-1:12]
	tmp2 = Atl3_1[j, 12 + monst + nstep+0:12 +monst + nstep + nlen * 12+0:12]
	tmp3 = Atl3_1[j, 12 + monst + nstep+1:12 +monst + nstep + nlen * 12+1:12]

	tmp   = (tmp1+tmp2+tmp3)/3.0

	Y[nlen * j:nlen * (j + 1), :] = tmp.reshape([nlen, 1])

	del [tmp1,tmp2,tmp3,tmp]


tmp1  = Atl3_2[12 + monst + nstep-1:12 + monst + nstep +nvali * 12-1:12]
tmp2  = Atl3_2[12 + monst + nstep+0:12 + monst + nstep +nvali * 12+0:12]
tmp3  = Atl3_2[12 + monst + nstep+1:12 + monst + nstep +nvali * 12+1:12]
tmp   = (tmp1+tmp2+tmp3)/3.0

YT[:] = tmp.reshape([nvali, 1])

del [tmp1,tmp2,tmp3,tmp]

print('X.shape', X.shape)
print('XT.shape', XT.shape)
print('Y.shape', Y.shape)
print('YT.shape', YT.shape)


# Build the model

model = models.Sequential()
model.add(layers.Conv2D(30, (4, 8),strides=(1, 1), use_bias=True, activation='tanh',  padding='SAME',
	kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
	bias_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01), input_shape=(nlat,nlon, 12)))
model.add(layers.MaxPooling2D((2, 2),strides=(2, 2), padding='SAME'))


model.add(layers.Conv2D(30, (2, 4),strides=(1, 1), use_bias=True, activation='tanh', padding='SAME',
	kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
	bias_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)))
model.add(layers.MaxPooling2D((2, 2),strides=(2, 2), padding='SAME'))


model.add(layers.Conv2D(30, (2, 4),strides=(1, 1), use_bias=True, activation='tanh', padding='SAME',
	kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
	bias_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)))

model.add(layers.Flatten())

model.add(layers.Dense(50, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
	bias_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)))
model.add(layers.Dense(1, activation=None,kernel_initializer=keras.initializers.RandomNormal(stddev=1.0),
	bias_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)))

model.summary() # 显示模型的架构


model.compile(optimizer=keras.optimizers.RMSprop(0.005,0.9), loss='mse')
#===========================================================================

EPOCH  = 100

model.fit(X, Y, epochs=EPOCH, batch_size=400)

YP       = model.predict(XT)
ACC 	 = np.corrcoef(YT.reshape(-1),YP.reshape(-1))[0][1]
sloss 	 = sp.sqrt(sp.mean(YT -YP)**2)
print('RSME: ', sloss, 'cor: ', ACC)


train_obj = 'Pred_Atl3_v1.0'

if (monst<12):
	start = monst+1
else:
	start = 1


dirpath = './' + train_obj + '_' + \
	str(start).zfill(2) + 's_' + \
	str(nstep + 1).zfill(2) + 'p_' + str(EPOCH).zfill(3) + 't'

if (not os.path.isdir(dirpath)):
	os.makedirs(dirpath)




logname = 'output_'+str(start).zfill(2) + 's_' + str(nstep + 1).zfill(2) + 'p_' + str(EPOCH).zfill(3) + 't.txt'

modelname = 'model_'+str(start).zfill(2) + 's_' + str(nstep + 1).zfill(2) + 'p_' + str(EPOCH).zfill(3) + 't.h5'

output = np.zeros((nvali,2),dtype=float)
output[:, 0] = YT[:, 0]
output[:, 1] = YP[:, 0]
np.savetxt(dirpath+'/'+logname,output,fmt='%10.5f')

model.save(dirpath+'/'+modelname)


import matplotlib.pyplot as plt


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5)) #width+height

ax.plot(YT[:, 0],'ko--')
ax.plot(YP[:, 0],'ro--')
plt.savefig('Prediction_Atl3_CNN_st-'+str(start).zfill(2)+'_lead-'+str(nstep+1).zfill(2)+'.svg',dpi=800,bbox_inches = 'tight')
