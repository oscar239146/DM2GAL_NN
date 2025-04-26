
from basemodule import *
from nnmodule import *
from NewFuncs import *
from sklearn.model_selection import train_test_split


d0 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/TNG_Cluster_Data.hdf5', 'r')
d1 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_train_plus.h5', 'r')
d2 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_test_plus.h5', 'r')
time = d1['time'][:]
age = d1['lookback_time'][:]

sims1 = d1["simu"][:]

sim = 0
if sim != 0:
    d1 = {key: d1[key] for key in d1.keys()}  
    for key in d1:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d1[key]
            x = x[sims1 == sim]
            d1[key] = x
    print("Filtered")

zero_arr = np.zeros(shape = d0['beta'][:].shape)

minMh = np.min(np.concatenate((d1['logMh'][:], d2['logMh'][:])))
Mh_filt = [(d0["logMh"][:]>=minMh)][0] #Filter halo masses as in HC23
#print(np.sum(Mh_filt), minMh)

Mh0 = d1["logMh"][:]
Ms0 = d1["logMs_or"][:]
Mh3 = d0["logMh"][Mh_filt]
Ms3 = d0["logMs"][Mh_filt]
#print(Mh_filt)

#keep_mask = SHMR_mask(Mh0, Ms0, Mh3, Ms3, sig = 2.5, n_bins = 10000)

keep_mask = SHMR_mask_high_bias(Mh0, Ms0, Mh3, Ms3, sig = 2.5, n_bins = 10000, bias_fact = 0.2)

print(np.sum(keep_mask))

#Mh_filt = np.where(keep_mask)[0] #Filtering very slow
print(Mh_filt)
del Mh0
del Mh3
del Ms0
del Ms3

#d0xt = np.stack((d0['Mhdot'][:], d0['delta_1'][:], d0['delta_3'][:], d0['delta_5'][:], d0['vcirc'][:], d0['rhalf'][:], d0['skew'][:], d0['minD'][:]), axis=-1)
#d0xs = np.stack((d0['beta'][:], zero_arr, zero_arr, zero_arr, zero_arr, zero_arr, d0['formtime'][:], d0['logMh'][:], d0['logmaxMhdot'][:]), axis=-1)
#d0y = np.concatenate((d0['SFH'][:], d0['ZH'][:], d0['logZ'][:].reshape(len(d0xs), 1), d0['logMs'][:].reshape(len(d0xs), 1), d0['mwsa'][:].reshape(len(d0xs), 1)), axis=-1)

zero_arr = zero_arr[Mh_filt]

d0xt = np.stack((d0['Mhdot'][Mh_filt], d0['delta_1'][Mh_filt], d0['delta_3'][Mh_filt], d0['delta_5'][Mh_filt], d0['vcirc'][Mh_filt], d0['rhalf'][Mh_filt], d0['skew'][Mh_filt], d0['minD'][Mh_filt]), axis=-1)
print("filt1")
d0xs = np.stack((d0['beta'][Mh_filt], zero_arr, zero_arr, zero_arr, zero_arr, zero_arr, d0['formtime'][Mh_filt], d0['logMh'][Mh_filt], d0['logmaxMhdot'][Mh_filt]), axis=-1)
print("filt2")
d0y = np.concatenate((d0['SFH'][Mh_filt], d0['ZH'][Mh_filt], d0['logZ'][Mh_filt].reshape(len(d0xs), 1), d0['logMs'][Mh_filt].reshape(len(d0xs), 1), d0['mwsa'][Mh_filt].reshape(len(d0xs), 1)), axis=-1)

print("Filtered")

d1xt = np.stack((d1['Mhdot'][:], d1['delta1'][:], d1['delta3'][:], d1['delta5'][:], d1['vcirc'][:], d1['rhalf'][:], d1['skew'][:], d1['minD'][:]), axis=-1)
d1xs = np.stack((d1['beta'][:], d1['d_min'][:], d1['d_node'][:], d1['d_saddle_1'][:], d1['d_saddle_2'][:], d1['d_skel'][:], d1['formtime'][:], d1['logMh'][:], d1['logmaxMhdot'][:]), axis=-1)
d1y = np.concatenate((d1['SFH'][:], d1['ZH'][:], d1['logZ'][:].reshape(len(d1xs), 1), d1['logMs_or'][:].reshape(len(d1xs), 1), d1['mwsa'][:].reshape(len(d1xs), 1)), axis=-1)

d2xt = np.stack((d2['Mhdot'][:], d2['delta1'][:], d2['delta3'][:], d2['delta5'][:], d2['vcirc'][:], d2['rhalf'][:], d2['skew'][:], d2['minD'][:]), axis=-1)
d2xs = np.stack((d2['beta'][:], d2['d_min'][:], d2['d_node'][:], d2['d_saddle_1'][:], d2['d_saddle_2'][:], d2['d_skel'][:], d2['formtime'][:], d2['logMh'][:], d2['logmaxMhdot'][:]), axis=-1)
d2y = np.concatenate((d2['SFH'][:], d2['ZH'][:], d2['logZ'][:].reshape(len(d2xs), 1), d2['logMs_or'][:].reshape(len(d2xs), 1), d2['mwsa'][:].reshape(len(d2xs), 1)), axis=-1)

#print(d1xt.shape, d1xs.shape, d1y.shape)
#d1xt = np.vstack((d1xt, d2xt))
#d1xs = np.vstack((d1xs, d2xs))
#d1y = np.vstack((d1y, d2y))
#print(d1xt.shape, d1xs.shape, d1y.shape)

#d0xt = np.vstack((d0xt, d1xt))
#d0xs = np.vstack((d0xs, d1xs))
#d0y = np.vstack((d0y, d1y))
#print("Stacked")

d0xt1q, d0xt1g, d0xt1o = GQTvecnorm(d0xt[:,:,0].reshape(len(d0xt), 33, 1))
d0xt2q, d0xt2g, d0xt2o = GQTscalnorm(d0xt[:,:,1:4])
d0xt3q, d0xt3g, d0xt3o = GQTvecnorm(d0xt[:,:,4:])
d0xtq = np.concatenate((d0xt1q, d0xt2q, d0xt3q), axis=-1)
d0xs1q, d0xs1g, d0xs1o = GQTscalnorm(d0xs[:,0].reshape(len(d0xs), 1))
d0xs2q = d0xs[:,1:6]
d0xs3q, d0xs3g, d0xs3o = GQTscalnorm(d0xs[:,6:])
d0xsq = np.concatenate((d0xs1q, d0xs2q, d0xs3q), axis=-1)
d0xq = [d0xtq, d0xsq]
d0y1q, d0y1g, d0y1o = GQTvecnorm(d0y[:,0:33].reshape(len(d0xs), 33, 1))
d0y2q, d0y2g, d0y2o = GQTvecnorm(d0y[:,33:66].reshape(len(d0xs), 33, 1))
d0y3q, d0y3g, d0y3o = GQTscalnorm(d0y[:,66:])
d0yq = np.concatenate((np.squeeze(d0y1q), np.squeeze(d0y2q), d0y3q), axis=-1)


#d0xt1q, d0xt1g, d0xt1o = GQTvecnorm(d0xt[:,:,0].reshape(len(d0xt), 33, 1))
#d0xt2q, d0xt2g, d0xt2o = GQTscalnorm(d0xt[:,:,1:4])
#d0xt3q, d0xt3g, d0xt3o = GQTvecnorm(d0xt[:,:,4:])
#d0xtq = np.concatenate((d0xt1q, d0xt2q, d0xt3q), axis=-1)
#d0xsq, d0xsg, d0xso = GQTscalnorm(d0xs)
#d0xq = [d0xtq, d0xsq]
#d0y1q, d0y1g, d0y1o = GQTvecnorm(d0y[:,0:33].reshape(len(d0xs), 33, 1))
#d0y2q, d0y2g, d0y2o = GQTvecnorm(d0y[:,33:66].reshape(len(d0xs), 33, 1))
#d0y3q, d0y3g, d0y3o = GQTscalnorm(d0y[:,66:])
#d0yq = np.concatenate((np.squeeze(d0y1q), np.squeeze(d0y2q), d0y3q), axis=-1)
print("Compiled Training Data")

#[8:14]
#d1xt1q, d1xt1g, d1xt1o = GQTvecnorm(d1xt[:,:,0].reshape(len(d1xt), 33, 1))
#d1xt2q, d1xt2g, d1xt2o = GQTscalnorm(d1xt[:,:,1:4])
#d1xt3q, d1xt3g, d1xt3o = GQTvecnorm(d1xt[:,:,4:])
#d1xtq = np.concatenate((d1xt1q, d1xt2q, d1xt3q), axis=-1)
#d1xsq, d1xsg, d1xso = GQTscalnorm(d1xs)
#d1xq = [d1xtq, d1xsq]
#d1y1q, d1y1g, d1y1o = GQTvecnorm(d1y[:,0:33].reshape(len(d1xs), 33, 1))
#d1y2q, d1y2g, d1y2o = GQTvecnorm(d1y[:,33:66].reshape(len(d1xs), 33, 1))
#d1y3q, d1y3g, d1y3o = GQTscalnorm(d1y[:,66:])
#d1yq = np.concatenate((np.squeeze(d1y1q), np.squeeze(d1y2q), d1y3q), axis=-1)


#print(d1y1g, d1y1o, d1y2g, d1y2o, d1y3g, d1y3o)
d2xt1q, d2xt1g, d2xt1o = GQTvecnorm(d2xt[:,:,0].reshape(len(d2xt), 33, 1))
d2xt2q, d2xt2g, d2xt2o = GQTscalnorm(d2xt[:,:,1:4])
d2xt3q, d2xt3g, d2xt3o = GQTvecnorm(d2xt[:,:,4:])
d2xtq = np.concatenate((d2xt1q, d2xt2q, d2xt3q), axis=-1)
d2xsq, d2xsg, d2xso = GQTscalnorm(d2xs)
d2xq = [d2xtq, d2xsq]
d2y1q, d2y1g, d2y1o = GQTvecnorm(d2y[:,0:33].reshape(len(d2xs), 33, 1))
d2y2q, d2y2g, d2y2o = GQTvecnorm(d2y[:,33:66].reshape(len(d2xs), 33, 1))
d2y3q, d2y3g, d2y3o = GQTscalnorm(d2y[:,66:])
d2yq = np.concatenate((np.squeeze(d2y1q), np.squeeze(d2y2q), d2y3q), axis=-1)

#sample_weights = np.ones(shape = (len(d0xs)))
#sw_TNG = np.ones(shape = (len(d1xs)))
#sw_TNG[sims1 == 100] = sw_TNG[sims1 == 100] * 0.5
#sample_weights = np.concatenate((keep_mask.astype(int), np.ones(shape = (len(d1xs)))))
sample_weights = keep_mask.astype(int)
#sample_weights = np.concatenate(((keep_mask.astype(int)+0.2)/1.1, sw_TNG))
#Give 0 weight to all samples with 2-sig different logMs, 0.6 weight to rest of samples

#sample_weights[:len(d0['beta'][Mh_filt])] = sample_weights[:len(d0['beta'][Mh_filt])]*0.3 #weight TNG-Cluster data as worth 30% of normal Bright data as around 6x more abundant
#print(sample_weights.shape)

model_name = "13_plus300"
#optimizer = keras.optimizers.RMSprop(0.0007)
optimizer = keras.optimizers.RMSprop(1e-6)
model = keras.models.load_model('central-model'+model_name+'.h5') 

#Freeze PreLu and their preceding dense layers
#for i in range(len(model.layers)):
#    if isinstance(model.layers[i], keras.layers.PReLU):
#        model.layers[i-1].trainable = False
#        model.layers[i].trainable = False

d0xtq, xt_val, d0xsq, xs_val, d0yq, y_val, w_train, w_val = train_test_split(d0xtq, d0xsq, d0yq, sample_weights, test_size=0.2, random_state=23)
d0xq = [d0xtq, d0xsq]
#CentralNetwork3()
model.compile(loss=tf.keras.losses.Huber(delta=0.6), optimizer=optimizer, metrics=['mse', 'mae', 'accuracy'], weighted_metrics = ['mse', 'mae'])
history = model.fit(d0xq, d0yq, epochs=3, validation_split=0.2, batch_size=16, verbose=1, callbacks=[early_stop_retrain, scheduler_retrain, lr_adjust_retrain], sample_weight = w_train, validation_data=([xt_val,xs_val], y_val, w_val))
#d3yq = model.predict(d2xq)
model.save('central-model'+model_name+'_Cluster_train_swmask1e6_3_retrained.h5')
d3yq = model.predict(d2xq)

d3y1 = np.squeeze(GQTvecinv(d3yq[:,0:33].reshape(len(d2xs), 33, 1), d2y1g, d2y1o))
d3y2 = np.squeeze(GQTvecinv(d3yq[:,33:66].reshape(len(d2xs), 33, 1), d2y2g, d2y2o))
d3y3 = GQTscalinv(d2y3q, d2y3g, d2y3o)
d3y = np.concatenate((d3y1, d3y2, d3y3), axis=-1)

X1 = 10**d1xs[:,-2]
X2 = 10**d2xs[:,-2]

M1 = sfhint(d1y[:,0:33], time=time)
M2 = sfhint(d2y[:,0:33], time=time)
M3 = sfhint(d3y[:,0:33], time=time)

Z1 = mwz(d1y[:,33:66], d1y[:,0:33], time=time)
Z2 = mwz(d2y[:,33:66], d2y[:,0:33], time=time)
Z3 = mwz(d3y[:,33:66], d3y[:,0:33], time=time)

A1 = mwa(age, d1y[:,0:33])
A2 = mwa(age, d2y[:,0:33])
A3 = mwa(age, d3y[:,0:33])

k = binmerger(X2, 10**np.linspace(np.log10(min(X2)), np.log10(max(X2)), 11))
kd = np.digitize(X2, k)
ck, cv, cw = count(kd)
a0 = np.asarray([10**np.median(np.log10(X2[o])) for o in cw])
a1 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
a1e = (a1-np.asarray([10**np.percentile(np.log10(M2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M2[o]), 85) for o in cw])-a1)
a2 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
a2e = (a2-np.asarray([10**np.percentile(np.log10(M3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M3[o]), 85) for o in cw])-a2)

plt.figure(figsize=(10, 6))
P1=plt.scatter(X2, M2, s=2, color='red')
P2=plt.scatter(X2, M3, s=2, color='blue')
P3=plt.scatter(a0, a1, s=40, color='darkred', marker='D')
plt.errorbar(a0, a1, a1e, ecolor='darkred', fmt='none')
P4=plt.scatter(a0, a2, s=40, color='darkblue', marker='D')
plt.errorbar(a0, a2, a2e, ecolor='darkblue', fmt='none')
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Halo Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylim(1e9, 2e13)
plt.xlim(2e10, 3e15)
plt.legend([(P1, P3), (P2, P4)], ['Raw Data', 'Predicted Data'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('shmr-central.pdf', bbox_inches = 'tight')
plt.close()

k = binmerger(M2, 10**np.linspace(np.log10(min(M2)), np.log10(max(M2)), 11))
kd = np.digitize(M2, k)
ck, cv, cw = count(kd)
b0 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
b1 = np.asarray([10**np.median(np.log10(Z2[o])) for o in cw])
b1e = (b1-np.asarray([10**np.percentile(np.log10(Z2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z2[o]), 85) for o in cw])-b1)
k = binmerger(M3, 10**np.linspace(np.log10(min(M3)), np.log10(max(M3)), 11))
kd = np.digitize(M3, k)
ck, cv, cw = count(kd)
c0 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
c1 = np.asarray([10**np.median(np.log10(Z3[o])) for o in cw])
c1e = (c1-np.asarray([10**np.percentile(np.log10(Z3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z3[o]), 85) for o in cw])-c1)

plt.figure(figsize=(10, 6))
P1=plt.scatter(M2, Z2, s=2, color='red')
P2=plt.scatter(M3, Z3, s=2, color='darkmagenta')
P3=plt.scatter(b0, b1, s=40, color='darkred', marker='D')
plt.errorbar(b0, b1, b1e, ecolor='darkred', fmt='none')
P4=plt.scatter(c0, c1, s=40, color='darkviolet', marker='D')
plt.errorbar(c0, c1, c1e, ecolor='darkviolet', fmt='none')
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'Mass-Weighted Metallicity ($Z_{\odot}$)', fontsize=18)
plt.ylim(0.1, 3)
plt.xlim(1e9, 1e13)
plt.legend([(P1, P3), (P2, P4)], ['Raw Data', 'Predicted Data'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('mzr-central.pdf', bbox_inches = 'tight')
plt.close()

digbins = np.percentile(M2, np.concatenate((np.linspace(0,90,10)[:-1], np.linspace(90, 100, 6))))
digbins[-1] *= 1.01
dig = np.digitize(M2, bins=digbins)
dk, dv, dw = count(dig)

ym = np.asarray([np.mean(M2[i]) for i in dw])
mwai = np.asarray([np.median(A2[i]) for i in dw])
mwaj = np.asarray([np.median(A3[j]) for j in dw])
mwais = np.asarray([iqr(A2[i]) for i in dw])
mwajs = np.asarray([iqr(A3[j]) for j in dw])

plt.scatter(ym, mwai, color='red', marker='D', label='True Data', zorder=2)
plt.errorbar(ym, mwai, yerr=mwais, fmt='none', ecolor='red', zorder=1)
plt.scatter(ym, mwaj, color='blue', marker='D', label='Predicted Data', zorder=2)
plt.errorbar(ym, mwaj, yerr=mwajs, fmt='none', ecolor='blue', zorder=1)
plt.xscale('log')
plt.xlim(1e9, 2e12)
plt.tick_params(axis='both', labelsize=12)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=16)
plt.ylabel('Stellar Mass Weighted Age (Gyr)', fontsize=16)
ylim=plt.gca().get_ylim()
plt.ylim(ylim)
plt.legend(loc=4, frameon=False, ncol=2)
plt.tight_layout()
plt.savefig('mwa-central.pdf', bbox_inches = 'tight')
plt.close()