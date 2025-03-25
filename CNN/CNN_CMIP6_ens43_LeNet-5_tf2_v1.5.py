"""
File: CNN_sst+zos.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/yourname
Description:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset

# import pylab as pl
import tensorflow as tf
# from keras import backend as K

import scipy      as sp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

#def conv_layer(prev_layer, numcore, runstep, is_training):
#    strides = runstep
#    conv_layer = tf.layers.conv2d(
#        prev_layer, numcore, kernel_size=(4, 8), strides=strides, padding='same', use_bias=False, activation='relu')
#    conv_layer = tf.layers.batch_normalization(
#        conv_layer, training=is_training)
#    conv_layer = tf.nn.relu(conv_layer)
#
#    return conv_layer
def conv_layer(prev_layer, numcore, runstep, is_training):
    strides = runstep
    conv_layer = tf.layers.conv2d(
        prev_layer, numcore, kernel_size=(4, 8), strides=strides, padding='same', use_bias=True, activation='tanh')
    conv_layer = tf.layers.batch_normalization(
        conv_layer, training=is_training)
    return conv_layer

def max_pool(x, kcore, runstep):
    return tf.nn.max_pool(x, ksize=[1, kcore, kcore, 1], strides=[1, runstep, runstep, 1], padding='SAME')


def fully_connected(prev_layer, num_units, is_training):
    layer = tf.layers.dense(prev_layer, num_units,
                            use_bias=True, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.tanh(layer)

    return layer


def fully_connected2(prev_layer, num_units, is_training):
    layer = tf.layers.dense(prev_layer, num_units,
                            use_bias=True, activation=None)
#    layer = tf.layers.batch_normalization(layer, training=is_training)
#    layer = tf.nn.leaky_relu(layer, alpha=0.56)

    return layer

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))

def get_batch(X, Y, n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys  # 生成每一个batch

##############################################################################
import math
 
# 函数：计算相关系数
def calc_corr(a, b):
    a_avg = sum(a)/len(a)
    b_avg = sum(b)/len(b)
 
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg)*(y - b_avg) for x,y in zip(a, b)])
 
    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg)**2 for x in a])*sum([(x - b_avg)**2 for x in b]))
 
    corr_factor = cov_ab/sq
 
    return corr_factor

##############################################################################
fsst1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_tos_historical_anom_detrend.nc"
fzos1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zos_historical_anom_detrend.nc"
fpsl1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zg_historical_anom_detrend_850hPa.nc"
fhgt1 = "/home/sunming/ML_WPSH/train_data/CMIP6set/CMIP6_zg_historical_anom_detrend_500hPa.nc"


fsst2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_ersstv5_198001-201812_Ham-LR.nc"
fzos2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_sshg_GODAS_198001-201812_Ham-LR.nc"
fpsl2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_hgt_JRA55_mon_198001-201812_850hPa_Ham-LR.nc"
fhgt2 = "/home/sunming/ML_WPSH/train_data/OBS_test/anom_detrend_hgt_JRA55_mon_198001-201812_500hPa_Ham-LR.nc"


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
# print(ssta1)
# monst = 1  # the last month we know
#monst = 10#int(os.environ.get("NMON"))
monst = int(os.environ.get("NMON"))
#nstep = 9#int(os.environ.get("NSTEP"))
nstep = int(os.environ.get("NSTEP"))

nstep = nstep - 1

print('nlat', nlat)
print('nlon', nlon)

id_lat = (lat >= 15) & (lat <= 25)
id_lon = (lon >= 110) & (lon <= 150)

nlen = 140
nens = 43
ntrain = nlen * nens
nvali = 36

# (150*3,lat(121),lon(240),3)
X = np.zeros((ntrain, nlat, nlon, 12), dtype=float)
Y = np.zeros((ntrain, 1), dtype=float)  # (150*3)

# (5*3,lat(121),lon(240),3)
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


#X[:, :, :, 3:] = X[:, :, :, 3:] * 10.0
#XT[:, :, :, 3:] = XT[:, :, :, 3:] * 10.0
print(psla1.shape)
for j in range(nens):
	tick1 = psla1[j, 12 + monst + nstep-1:12 +monst + nstep + nlen * 12-1:12, :,:]
	tick2 = psla1[j, 12 + monst + nstep+0:12 +monst + nstep + nlen * 12+0:12, :,:]
	tick3 = psla1[j, 12 + monst + nstep+1:12 +monst + nstep + nlen * 12+1:12, :,:]

	tmp11 = np.mean(tick1[:, id_lat,:], axis=1) #.reshape([nlen, 1])
	tmp1  = np.mean(tmp11[:,id_lon],axis=1)
	tmp21 = np.mean(tick2[:, id_lat,:], axis=1) #.reshape([nlen, 1])
	tmp2  = np.mean(tmp21[:,id_lon],axis=1)
	tmp31 = np.mean(tick3[:, id_lat,:], axis=1) #.reshape([nlen, 1])
	tmp3  = np.mean(tmp31[:,id_lon],axis=1)

	tmp   = (tmp1+tmp2+tmp3)/3.0

	Y[nlen * j:nlen * (j + 1), :] = tmp.reshape([nlen, 1])

	del [tmp11,tmp21,tmp31,tmp1,tmp2,tmp3,tmp,tick1,tick2,tick3]


tick1  = psla2[12 + monst + nstep-1:12 + monst + nstep +nvali * 12-1:12, :, :]
tick2  = psla2[12 + monst + nstep+0:12 + monst + nstep +nvali * 12+0:12, :, :]
tick3  = psla2[12 + monst + nstep+1:12 + monst + nstep +nvali * 12+1:12, :, :]

tmp11 = np.mean(tick1[:, id_lat,:], axis=1) #.reshape([nlen, 1])
tmp1  = np.mean(tmp11[:,id_lon],axis=1)
tmp21 = np.mean(tick2[:, id_lat,:], axis=1) #.reshape([nlen, 1])
tmp2  = np.mean(tmp21[:,id_lon],axis=1)
tmp31 = np.mean(tick3[:, id_lat,:], axis=1) #.reshape([nlen, 1])
tmp3  = np.mean(tmp31[:,id_lon],axis=1)

tmp   = (tmp1+tmp2+tmp3)/3.0

YT[:] = tmp.reshape([nvali, 1])

del [tmp11,tmp21,tmp31,tmp1,tmp2,tmp3,tmp,tick1,tick2,tick3]


Y  = Y/10.0
YT = YT/10.0


print('X.shape', X.shape)
print('XT.shape', XT.shape)
print('Y.shape', Y.shape)
print('YT.shape', YT.shape)

#print(Y)
#print(YT)





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

#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

EPOCH = 100


model.compile(optimizer=keras.optimizers.RMSprop(0.005,0.9), loss='mse')

model.fit(X, Y, epochs=EPOCH, batch_size=400)

#Tloss,acc = model.evaluate(XT, YT)
YP       = model.predict(XT)

#print('loss is ',Tloss ,'and acc is',acc)
#print(YP)

tmpor = calc_corr(YT,YP)
sloss = sp.sqrt(sp.mean(YT -YP)**2)
print('RSME: ', sloss, 'cor: ', tmpor)


train_obj = 'Pred_PSACv1.0'

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
#
#
#
#
#
#
#
## ===================== build the structure of CNN===================================
#xs = tf.placeholder(tf.float32, [None, nlat, nlon, 6])  # 28x28
#ys = tf.placeholder(tf.float32, [None, 1])
## keep prob is the parameter for drop out
#
#
#num_convf = 30
#num_hiddf = 50
#xdim2 = int(nlon/4)
#ydim2 = int(nlat/4)
#
#
#w = init_weights([8, 4, 6, num_convf])
#b = init_bias([num_convf])
#w2 = init_weights([4, 2, num_convf, num_convf])
#b2 = init_bias([num_convf])
#w3 = init_weights([4, 2, num_convf, num_convf])
#b3 = init_bias([num_convf])
#w4 = init_weights([num_convf * xdim2 * ydim2, num_hiddf])
#b4 = init_bias([num_hiddf])
#w_o = init_weights([num_hiddf, 1])
#b_o = init_bias([1])
#
#
#l1a = tf.tanh(tf.nn.conv2d(xs, w, strides=[1, 1, 1, 1], padding='SAME') + b)
#l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
##l1 = tf.nn.dropout(l1, p_keep_conv)
#
#l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
#l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
##l2 = tf.nn.dropout(l2, p_keep_conv)
#
#l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
#l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])
##l3 = tf.nn.dropout(l3, p_keep_conv)
#
#
#l4 = tf.tanh(tf.matmul(l3, w4) + b4)
##l4 = tf.nn.dropout(l4, p_keep_hidden)
#
#prediction = tf.matmul(l4, w_o) + b_o
#
#
##----------------------------------------------------
#keep_prob = tf.placeholder(tf.float32)
#is_training = tf.placeholder(tf.bool)
#
##==============================================================
##loss = tf.reduce_mean(tf.square(ys - prediction))
#loss = tf.reduce_mean(tf.squared_difference(prediction, ys))
#
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#    #train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
#    train_step = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(loss)
#
#
#saver = tf.train.Saver()   # set the container for net variables
#
#
#tmp = np.zeros((20 * 2, nvali), dtype=float)
#
##############################################################
#train_obj = 'CMIP6_ens43'
#
#
#for k in range(nstep, nstep + 1):
#    print('the object mon is ', monst + k + 1)
#
#    sess = tf.Session()
#    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#        init = tf.initialize_all_variables()
#    else:
#        init = tf.global_variables_initializer()
#    sess.run(init)
#
#    # record cost
#    cost = []
#    cor = []
#    yprec = []
#    record_step = 5
#    yprec.append(YT)
#
#    for i in range(201):
#        for batch_xs, batch_ys in get_batch(X, Y, Y.shape[0], batch_size=400):
#            sess.run(train_step, feed_dict={
#                     xs: batch_xs, ys: batch_ys, keep_prob: 0.9, is_training: True})
#        if i % record_step == 0:
#            print('this is the ', i, 'th train')
#            # print(YT)
#            sloss, yp = sess.run([loss, prediction], feed_dict={
#                xs: XT, ys: YT, keep_prob: 1.0, is_training: False})
#            cost.append(sloss)
#            yprec.append(yp)
#            rack1 = np.array(YT,dtype='float')
#            rack2 = np.array(yp,dtype='float')
#            #tmpor = pearsonr(rack1,rack2)[0]
#            tmpor = calc_corr(rack1,rack2)
#            print('RSME: ', sloss, 'cor: ', tmpor)
#            flog = open("./log_" + train_obj + "_" + str(monst).zfill(2) +
#                        "_" + str(k + 1).zfill(2) + ".txt", 'a+')
#            print('RSME: ', sloss, 'cor: ', tmpor, file=flog)
#            flog.close()
#            cor.append(tmpor)
#
##    print(cor)
#
#    dirpath = './' + train_obj + '_' + \
#        str(monst).zfill(2) + 's_' + \
#        str(k + 1).zfill(2) + 'p_' + str(i).zfill(3) + 't'
#    print(dirpath)
#    if (not os.path.isdir(dirpath)):
#        os.makedirs(dirpath)
#    save_path = saver.save(sess, dirpath + "/save_net.ckpt")
#    print("Save to path: ", save_path)
#    tmp[k * 2, :] = YT[:, 0]
#    tmp[k * 2 + 1, :] = yp[:, 0]
#    sess.close()
#    DIR = pd.DataFrame(tmp)
#    DIR.to_csv(str(monst).zfill(2) + 's_' +
#               str(k + 1).zfill(2) + 'p_' + str(i).zfill(3) + 't.csv')
