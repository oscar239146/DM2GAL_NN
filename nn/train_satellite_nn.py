from basemodule import *
from nnmodule import *

d1 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_nndata_train_plus.h5', 'r')
d2 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_nndata_test_plus.h5', 'r')
time = d1['time'][:]
age = d1['lookback_time'][:]

sims1 = d1["simu"][:]
sims2 = d2["simu"][:]

d1 = {key: d1[key] for key in d1.keys()}  
d2 = {key: d2[key] for key in d2.keys()}  

#Change to reflect which sim wanted
sim = 0
if sim != 0:
    for key in d1:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d1[key]
            x = x[sims1 == sim]
            d1[key] = x
    for key in d2:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d2[key]
            x = x[sims2 == sim]
            d2[key] = x


d1xt = np.stack((d1['Mhdot'][:], d1['mhdot'][:], d1['delta'][:], d1['vcirc'][:], d1['rhalf'][:], d1['skew'][:], d1['minD'][:]), axis=-1)
d1xs = np.stack((d1['beta_halo'][:], d1['beta_sub'][:], d1['a_inf'][:], d1['a_max'][:], d1['logMu'][:], d1['logVrel'][:], d1['formtime'][:], d1['logMh'][:], d1['logmaxMhdot'][:], d1['logmh'][:], d1['logmaxmhdot'][:]), axis=-1)
d1y = np.concatenate((d1['SFH'][:], d1['ZH'][:], d1['logZ'][:].reshape(len(d1xs), 1), d1['logMs'][:].reshape(len(d1xs), 1), d1['mwsa'][:].reshape(len(d1xs), 1)), axis=-1)

d2xt = np.stack((d2['Mhdot'][:], d2['mhdot'][:], d2['delta'][:], d2['vcirc'][:], d2['rhalf'][:], d2['skew'][:], d2['minD'][:]), axis=-1)
d2xs = np.stack((d2['beta_halo'][:], d2['beta_sub'][:], d2['a_inf'][:], d2['a_max'][:], d2['logMu'][:], d2['logVrel'][:], d2['formtime'][:], d2['logMh'][:], d2['logmaxMhdot'][:], d2['logmh'][:], d2['logmaxmhdot'][:]), axis=-1)
d2y = np.concatenate((d2['SFH'][:], d2['ZH'][:], d2['logZ'][:].reshape(len(d2xs), 1), d2['logMs'][:].reshape(len(d2xs), 1), d2['mwsa'][:].reshape(len(d2xs), 1)), axis=-1)

d1xt1q, d1xt1g, d1xt1o = GQTvecnorm(d1xt[:,:,0:2])
d1xt2q, d1xt2g, d1xt2o = GQTscalnorm(d1xt[:,:,2].reshape(len(d1xt), 33, 1))
d1xt3q, d1xt3g, d1xt3o = GQTvecnorm(d1xt[:,:,3:])
d1xtq = np.concatenate((d1xt1q, d1xt2q, d1xt3q), axis=-1)
d1xsq, d1xsg, d1xso = GQTscalnorm(d1xs)
d1xq = [d1xtq, d1xsq]
d1y1q, d1y1g, d1y1o = GQTvecnorm(d1y[:,0:33].reshape(len(d1xs), 33, 1))
d1y2q, d1y2g, d1y2o = GQTvecnorm(d1y[:,33:66].reshape(len(d1xs), 33, 1))
d1y3q, d1y3g, d1y3o = GQTscalnorm(d1y[:,66:])
d1yq = np.concatenate((np.squeeze(d1y1q), np.squeeze(d1y2q), d1y3q), axis=-1)

d2xt1q, d2xt1g, d2xt1o = GQTvecnorm(d2xt[:,:,0:2])
d2xt2q, d2xt2g, d2xt2o = GQTscalnorm(d2xt[:,:,2].reshape(len(d2xt), 33, 1))
d2xt3q, d2xt3g, d2xt3o = GQTvecnorm(d2xt[:,:,3:])
d2xtq = np.concatenate((d2xt1q, d2xt2q, d2xt3q), axis=-1)
d2xsq, d2xsg, d2xso = GQTscalnorm(d2xs)
d2xq = [d2xtq, d2xsq]
d2y1q, d2y1g, d2y1o = GQTvecnorm(d2y[:,0:33].reshape(len(d2xs), 33, 1))
d2y2q, d2y2g, d2y2o = GQTvecnorm(d2y[:,33:66].reshape(len(d2xs), 33, 1))
d2y3q, d2y3g, d2y3o = GQTscalnorm(d2y[:,66:])
d2yq = np.concatenate((np.squeeze(d2y1q), np.squeeze(d2y2q), d2y3q), axis=-1)

model = SatelliteNetwork2()
history = model.fit(d1xq, d1yq, epochs=EPOCHS, batch_size=20, validation_split=0.1, verbose=1, callbacks=[early_stop, scheduler, lr_adjust])
d3yq = model.predict(d2xq)
model.save('satellite-model12_2.h5')
#batch size 20, val split 0.1 used for original model
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
plt.ylim(1e9, 1e13)
plt.xlim(1e9, 3e14)
plt.legend([(P1, P3), (P2, P4)], ['Raw Data', 'Predicted Data'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('shmr-satellite.pdf', bbox_inches = 'tight')
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
plt.ylim(0.01, 10)
plt.xlim(1e9, 1e13)
plt.legend([(P1, P3), (P2, P4)], ['Raw Data', 'Predicted Data'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('mzr-satellite.pdf', bbox_inches = 'tight')
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
plt.savefig('mwa-satellite.pdf', bbox_inches = 'tight')
plt.close()