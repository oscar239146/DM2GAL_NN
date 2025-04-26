#%%
from basemodule import *
from nnmodule import *
from NewFuncs import *


c0_2 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/TNG_Cluster_Data.hdf5', 'r')
d0 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_train_plus.h5', 'r')
d1 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_test_plus.h5', 'r')
d2 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_pred.h5', 'r')
d3 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_dark.h5', 'r')
model = keras.models.load_model('central-model13_plus300.h5')
model_C = keras.models.load_model('central-model13_plus_Cluster_train_swmask1e6_5_retrained.h5')
time = d3['time'][:] 
age = d3['lookback_time'][:]

sims0 = d0["simu"][:]
sims1 = d1["simu"][:]
sims2 = d2["simu"][:]
sims3 = d3["simu"][:]
#%%
print(len(sims3==300), len(sims3==100))
print(len(sims0==300), len(sims0==100))
mind0Mh = np.min(d0["logMh"][sims0==300])
print(mind0Mh)
fig, ax = plt.subplots(3,1, sharex=True, sharey=True, constrained_layout=True)
ax[0].hist2d(d0["logMh"][sims0==100], d0["logMs_or"][sims0==100], bins=100)
ax[1].hist2d(d0["logMh"][sims0==300], d0["logMs_or"][sims0==300], bins=100)
ax[2].hist2d(c0_2["logMh"][c0_2["logMh"]>=mind0Mh], c0_2["logMs"][c0_2["logMh"]>=mind0Mh], bins=100)
fig.close()
#ax[0].set_xlim(10.5, 15)
#%%
#Save copies to memory
d0 = {key: d0[key] for key in d0.keys()}  
d1 = {key: d1[key] for key in d1.keys()}  
d2 = {key: d2[key] for key in d2.keys()}  
d3 = {key: d3[key] for key in d3.keys()}  

#Mh3 = d3['logMh'][:]
#Mh0 = d0['logMh'][sims0 == 300]
#minMh0 = np.min(Mh0)

#for key in d3:
#        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
#            x = d3[key]
#            x = x[Mh3 > minMh0]
#            d3[key] = x

d3xt = np.stack((d3['Mhdot'][:], d3['delta1'][:], d3['delta3'][:], d3['delta5'][:], d3['vcirc'][:], d3['rhalf'][:], d3['skew'][:], d3['minD'][:]), axis=-1) #Temporal quantities
d3xs = np.stack((d3['beta'][:], d3['d_min'][:], d3['d_node'][:], d3['d_saddle_1'][:], d3['d_saddle_2'][:], d3['d_skel'][:], d3['formtime'][:], d3['logMh'][:], d3['logmaxMhdot'][:]), axis=-1) #Non-temporal quantities

d1xt = np.stack((d0['Mhdot'][:], d0['delta1'][:], d0['delta3'][:], d0['delta5'][:], d0['vcirc'][:], d0['rhalf'][:], d0['skew'][:], d0['minD'][:]), axis=-1)
d1xs = np.stack((d0['beta'][:], d0['d_min'][:], d0['d_node'][:], d0['d_saddle_1'][:], d0['d_saddle_2'][:], d0['d_skel'][:], d0['formtime'][:], d0['logMh'][:], d0['logmaxMhdot'][:]), axis=-1)
d1y = np.concatenate((d0['SFH'][:], d0['ZH'][:], d0['logZ'][:].reshape(len(d1xs), 1), d0['logMs'][:].reshape(len(d1xs), 1), d0['mwsa'][:].reshape(len(d1xs), 1)), axis=-1)

d2xt = np.stack((d1['Mhdot'][:], d1['delta1'][:], d1['delta3'][:], d1['delta5'][:], d1['vcirc'][:], d1['rhalf'][:], d1['skew'][:], d1['minD'][:]), axis=-1)
d2xs = np.stack((d1['beta'][:], d1['d_min'][:], d1['d_node'][:], d1['d_saddle_1'][:], d1['d_saddle_2'][:], d1['d_skel'][:], d1['formtime'][:], d1['logMh'][:], d1['logmaxMhdot'][:]), axis=-1)
d2y = np.concatenate((d1['SFH'][:], d1['ZH'][:], d1['logZ'][:].reshape(len(d2xs), 1), d1['logMs'][:].reshape(len(d2xs), 1), d1['mwsa'][:].reshape(len(d2xs), 1)), axis=-1)

#final_layer = model.layers[-1]

# Get current bias
#current_bias = final_layer.get_weights()[1]



#adjusted_bias = current_bias+0.1 

#weights = final_layer.get_weights()
#weights[1] = adjusted_bias
#final_layer.set_weights(weights)


SFH0 = np.concatenate((d0['SFH'][:], d1['SFH'][:]), axis=0)
#SFH0 = d0['SFH'][:]
SFHq, SFHg, SFHo = GQTvecnorm(SFH0.reshape(*np.shape(SFH0), 1)) #added offsets to output
#%%
#model.layers[-1].trainable
#SFHq, SFHg, SFHo = GQTvecnorm(SFH0.reshape(len(d1xs),33, 1))
#plt.plot(SFHq[:,32,0])
#SFHo[0,0] = 4.12431804
#SFHo[0,1] = 0.95808776
#print(len(d1xs), len(SFH0), len(d3xs))

#plt.hist(10**d0["logMS"][:],histtype="step", log=True, density=True, color="blue")
#plt.hist(10**d3["logMS"][:],histtype="step", log=True, density=True, color="red")
#plt.hist(M3,histtype="step", log=True, density=True, color="green")
#plt.title("Residuals")
#plt.scatter(10**d3["logMS"][:], 10**d3["logMS"][:]-M3, s=1)
#plt.show()
#plt.close()
#plt.hist(10**d0["logMh"][:],histtype="step", log=True, density=True, color="blue")
#plt.hist(10**d3["logMh"][:],histtype="step", log=True, density=True, color="red")
#plt.hist(10**d1["logMh"][:],histtype="step", log=True, density=True, color="green")

#plt.xscale("log")
#%%
ZH0 = np.concatenate((d0['ZH'][:], d1['ZH'][:]), axis=0)
ZHq, ZHg, ZHo = GQTvecnorm(ZH0.reshape(*np.shape(ZH0), 1))

#others0 = np.concatenate((np.concatenate((d0['logZ'][:], d1['logZ'][:])).reshape(len(ZH0), 1),
#    np.concatenate((d0['logMs'][:],d1['logMs'][:])).reshape(len(ZH0), 1),
#    np.concatenate((d0['mwsa'][:],d1['mwsa'][:])).reshape(len(ZH0), 1)), axis=-1)
others1 = np.concatenate((c0_2['logZ'][:].reshape(len(c0_2['logZ'][:]), 1), c0_2['logMs'][:].reshape(len(c0_2['logZ'][:]), 1), c0_2['mwsa'][:].reshape(len(c0_2['logZ'][:]), 1)), axis=-1)

others2 = np.concatenate((d1['logZ'][:].reshape(len(d2xs), 1), d1['logMs'][:].reshape(len(d2xs), 1), d1['mwsa'][:].reshape(len(d2xs), 1)), axis=-1)

others0 = np.concatenate((d0['logZ'][:].reshape(len(d1xs), 1), d0['logMs'][:].reshape(len(d1xs), 1), d0['mwsa'][:].reshape(len(d1xs), 1)), axis=-1)

others0 = np.vstack((others0, others2))#np.vstack((others1, others0))
othq, othg, otho = GQTscalnorm(others0)

d3xt1q, d3xt1g = GQTvecnorm(d3xt[:,:,0].reshape(len(d3xt), 33, 1))[:2] #output only first 2 and reshape
d3xt2q, d3xt2g = GQTscalnorm(d3xt[:,:,1:4])[:2]
d3xt3q, d3xt3g = GQTvecnorm(d3xt[:,:,4:])[:2]

d3xtq = np.concatenate((d3xt1q, d3xt2q, d3xt3q), axis=-1)
d3xsq, d3xsg, d3xso = GQTscalnorm(d3xs)
d3xq = [d3xtq, d3xsq]
d3yq = model.predict(d3xq)
d3yq_C = model_C.predict(d3xq)
#%%
#sims3 = d3["simu"][:]
#Change to reflect which sim wanted (Use 0 for both sims)
sim = 300
if sim != 0:
    for key in d0:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d0[key]
            x = x[sims0 == sim]
            d0[key] = x
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
    for key in d3:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d3[key]
            x = x[sims3 == sim]
            d3[key] = x
#%%
SFH = np.squeeze(GQTvecinv(d3yq[:,0:33].reshape(len(d3xs), 33, 1), SFHg, SFHo)) #add offsets
ZH = np.squeeze(GQTvecinv(d3yq[:,33:66].reshape(len(d3xs), 33, 1), ZHg, ZHo))

others = np.squeeze(GQTscalinv(d3yq[:,66:], othg, otho))
others_C = np.squeeze(GQTscalinv(d3yq_C[:,66:], othg, otho))

if sim !=0:
    others = others[sims3 == sim]
    others_C = others_C[sims3 == sim]
    ZH = ZH[sims3 == sim]
    SFH = SFH[sims3 == sim]

#%%
X1 = 10**d1['logMh'][:]
X2 = 10**d3['logMh'][:]
X3 = 10**d3['logMh'][:]
X0 = 10**d0['logMh'][:]
plt.title("Halo Mass Probability Distribution")
plt.hist(X1,histtype="step", log=True, density=True, color="blue", label = "Training")
plt.hist(X0,histtype="step", log=True, density=True, color="red", label = "Testing")
plt.hist(X3,histtype="step", log=True, density=True, color="green", label="Dark")
plt.xscale("log")
plt.legend()
plt.close()
#%%
M1 = d1['logMs_or'][:]
M2 = d3['logMs'][:]
M3 = 10**others[:,1]
M3_C = 10**others_C[:,1]
#M1 = d1['logMS'][:]
#M2 = d2['logMS'][:]
#M3 = sfhint(SFH, time=time)

plt.hist(10**d0["logMs"][:],histtype="step", log=True, density=True, color="blue")
plt.hist(10**d3["logMs"][:],histtype="step", log=True, density=True, color="red")
plt.hist(M3,histtype="step", log=True, density=True, color="green")
plt.xscale("log")
#print(m_bins)
plt.close()
#for i in range(0,n_bins):
#    if (m_bins[i+1]<1.5e+12):
#        M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] = M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] * m_factors[i]*1.9
#    elif (m_bins[i+1]<2e+12):
#        M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] = M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] * m_factors[i]*1.4       
#    else:
#        M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] = M3[(X3>=m_bins[i])&(X3<m_bins[i+1])] * m_factors[i]


#Z1 = d1['logZS'][:]
#Z2 = d2['logZS'][:]
#Z3 = mwz(ZH, SFH, time=time)

Z1 = d1['logZ'][:]
Z2 = d3['logZ'][:]
Z3 = 10**others[:,0]
Z3_C = 10**others_C[:,0]

A1 = mwa(age, d1['SFH'][:])
A2 = mwa(age, d3['SFH'][:])
A3 = mwa(age, SFH)


M1, M2 = 10**M1, 10**M2 #Convert to non logged values
Z1, Z2 = 10**Z1, 10**Z2

#%%
k = binmerger(X1, 10**np.linspace(np.log10(min(X1)), np.log10(max(X1)), 11))
kd = np.digitize(X1, k)
ck, cv, cw = count(kd)
a0 = np.asarray([10**np.median(np.log10(X1[o])) for o in cw])
a1 = np.asarray([10**np.median(np.log10(M1[o])) for o in cw])
a1e = (a1-np.asarray([10**np.percentile(np.log10(M1[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M1[o]), 85) for o in cw])-a1)

k = binmerger(X3, 10**np.linspace(np.log10(min(X3)), np.log10(max(X3)), 11))
kd = np.digitize(X3, k)
ck, cv, cw = count(kd)
b0 = np.asarray([10**np.median(np.log10(X3[o])) for o in cw])
b2 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
b2e = (b2-np.asarray([10**np.percentile(np.log10(M3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M3[o]), 85) for o in cw])-b2)

k = binmerger(X3, 10**np.linspace(np.log10(min(X3)), np.log10(max(X3)), 11))
kd = np.digitize(X3, k)
ck, cv, cw = count(kd)
C0 = np.asarray([10**np.median(np.log10(X3[o])) for o in cw])
C2 = np.asarray([10**np.median(np.log10(M3_C[o])) for o in cw])
C2e = (C2-np.asarray([10**np.percentile(np.log10(M3_C[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M3_C[o]), 85) for o in cw])-C2)

k = binmerger(X2, 10**np.linspace(np.log10(min(X2)), np.log10(max(X2)), 11))
kd = np.digitize(X2, k)
ck, cv, cw = count(kd)
a2_0 = np.asarray([10**np.median(np.log10(X2[o])) for o in cw])
a2 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
a2e = (a2-np.asarray([10**np.percentile(np.log10(M2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(M2[o]), 85) for o in cw])-a2)

plt.figure(figsize=(10, 6))
P1=plt.scatter(X1, M1, s=0.1, color='blue', zorder=11)
P2=plt.scatter(X2, M2, s=0.1, color='red', zorder=-1)
P3=plt.scatter(X3, M3, s=0.1, color='green', zorder=12)
P7=plt.scatter(X3, M3_C, s=0.1, color='black')

P4=plt.scatter(a0, a1, s=40, color='navy', marker='D', zorder=13)
plt.errorbar(a0, a1, a1e, ecolor='navy', fmt='none', zorder=13)
P5=plt.scatter(a2_0, a2, s=40, color='maroon', marker='D', zorder=12)
plt.errorbar(a2_0, a2, a2e, ecolor='maroon', fmt='none', zorder=12)
P6=plt.scatter(b0, b2, s=40, color='limegreen', marker='D', zorder=14)
plt.errorbar(b0, b2, b2e, ecolor='limegreen', fmt='none', zorder=14)
P8=plt.scatter(C0, C2, s=40, color='yellow', marker='D', zorder=12)
plt.errorbar(C0, C2, C2e, ecolor='yellow', fmt='none', zorder=12)

plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Halo Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylim(1e9, 2e13)
plt.xlim(2e10, 3e15)
plt.title("TNG"+str(sim)+" SHMR", fontsize=18)
plt.legend([(P1, P4), (P2, P5), (P3, P6), (P7, P8)], ['TNG-Bright', 'HC24 NN', 'Modified NN', 'w/ TNG-Cluster'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2), P3:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
#plt.savefig('shmr-central-dark_'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('shmr-central-dark_plus'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('shmr-central-dark.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
#%%
k = binmerger(M1, 10**np.linspace(np.log10(min(M1)), np.log10(max(M1)), 11))
kd = np.digitize(M1, k)
ck, cv, cw = count(kd)
b0 = np.asarray([10**np.median(np.log10(M1[o])) for o in cw])
b1 = np.asarray([10**np.median(np.log10(Z1[o])) for o in cw])
b1e = (b1-np.asarray([10**np.percentile(np.log10(Z1[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z1[o]), 85) for o in cw])-b1)

k = binmerger(M2, 10**np.linspace(np.log10(min(M2)), np.log10(max(M2)), 11))
kd = np.digitize(M2, k)
ck, cv, cw = count(kd)
c0 = np.asarray([10**np.median(np.log10(M2[o])) for o in cw])
c1 = np.asarray([10**np.median(np.log10(Z2[o])) for o in cw])
c1e = (c1-np.asarray([10**np.percentile(np.log10(Z2[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z2[o]), 85) for o in cw])-c1)

k = binmerger(M3, 10**np.linspace(np.log10(min(M3)), np.log10(max(M3)), 11))
kd = np.digitize(M3, k)
ck, cv, cw = count(kd)
d0_2 = np.asarray([10**np.median(np.log10(M3[o])) for o in cw])
d1_2 = np.asarray([10**np.median(np.log10(Z3[o])) for o in cw])
d1e = (d1_2-np.asarray([10**np.percentile(np.log10(Z3[o]), 15) for o in cw]), np.asarray([10**np.percentile(np.log10(Z3[o]), 85) for o in cw])-d1_2)

plt.figure(figsize=(10, 6))
P1=plt.scatter(M1, Z1, s=1, color='red', zorder=1)
P2=plt.scatter(M2, Z2, s=1, color='darkmagenta', zorder=3)
P3=plt.scatter(M3, Z3, s=1, color='teal', zorder=2)
P4=plt.scatter(b0, b1, s=40, color='darkred', marker='D', zorder=4)
plt.errorbar(b0, b1, b1e, ecolor='darkred', fmt='none', zorder=4)
P5=plt.scatter(c0, c1, s=40, color='darkviolet', marker='D', zorder=6)
plt.errorbar(c0, c1, c1e, ecolor='darkviolet', fmt='none', zorder=6)
P6=plt.scatter(d0_2, d1_2, s=40, color='deepskyblue', marker='D', zorder=5)
plt.errorbar(d0_2, d1_2, d1e, ecolor='deepskyblue', fmt='none', zorder=5)
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=18)
plt.ylabel(r'NN Predicted Metallicity ($Z_{\odot}$)', fontsize=18)
plt.ylim(0.1, 3)
plt.xlim(1e9, 1e13)
plt.legend([(P1, P4), (P2, P5), (P3, P6)], ['TNG-Bright', 'HC24 NN', 'Modified NN'], handler_map={P1:leghand(numpoints=2), P2:leghand(numpoints=2), P3:leghand(numpoints=2)}, loc=2, frameon=False, fontsize=14)
plt.tight_layout()
#plt.savefig('mzr-central-dark_'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('mzr-central-dark_plus'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('mzr-central-dark.pdf', bbox_inches = 'tight')
plt.show()
plt.close()
#%%
digbins = np.percentile(M1, np.concatenate((np.linspace(0,90,10)[:-1], np.linspace(90, 100, 6))))
digbins[-1] *= 1.01
dig = np.digitize(M1, bins=digbins)
dk, dv, dw = count(dig)

ym = np.asarray([np.mean(M1[i]) for i in dw])
mwai = np.asarray([np.median(A1[i]) for i in dw])

mwais = np.asarray([iqr(A1[i]) for i in dw])


digbins = np.percentile(M3, np.concatenate((np.linspace(0,90,10)[:-1], np.linspace(90, 100, 6))))
digbins[-1] *= 1.01
dig = np.digitize(M3, bins=digbins)
dk, dv, dw = count(dig)

ym2 = np.asarray([np.mean(M2[i]) for i in dw])
Ym = np.asarray([np.mean(M3[i]) for i in dw])
Mwai = np.asarray([np.median(A3[i]) for i in dw])
Mwais = np.asarray([iqr(A3[i]) for i in dw])
mwaj = np.asarray([np.median(A2[j]) for j in dw])
mwajs = np.asarray([iqr(A2[j]) for j in dw])

plt.scatter(ym, mwai, color='blue', marker='D', label='TNG-Bright', zorder=2)
plt.errorbar(ym, mwai, yerr=mwais, fmt='none', ecolor='blue', zorder=1)
plt.scatter(ym2, mwaj, color='red', marker='D', label='HC24 NN', zorder=2)
plt.errorbar(ym2, mwaj, yerr=mwajs, fmt='none', ecolor='red', zorder=1)
plt.scatter(Ym, Mwai, color='green', marker='D', label='Modified NN', zorder=2)
plt.errorbar(Ym, Mwai, yerr=Mwais, fmt='none', ecolor='green', zorder=1) #Says Mwajs before
plt.xscale('log')
plt.xlim(1e9, 2e12)
plt.tick_params(axis='both', labelsize=12)
plt.xlabel(r'Stellar Mass ($M_{\odot}$)', fontsize=16)
plt.ylabel('Stellar Mass Weighted Age (Gyr)', fontsize=16)
ylim=plt.gca().get_ylim()
plt.ylim(ylim)
plt.legend(loc=4, frameon=False)
plt.tight_layout()
#plt.savefig('mwa-central-dark_'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('mwa-central-dark_plus'+str(sim)+'.pdf', bbox_inches = 'tight')
#plt.savefig('mwa-central-dark.pdf', bbox_inches = 'tight')
plt.show()
plt.close()

# %%
ran_ind = int(np.random.rand(1)*len(SFH))
print( ran_ind)
     
plt.plot(SFH[ran_ind])
plt.plot(d3["SFH"][ran_ind])
plt.plot(d0["SFH"][ran_ind])
plt.yscale("log")
#%%

minMh = np.min((X1, X2))
Mh0 = d0["logMh"][:]
Ms0 = d0["logMs_or"][:]
Mh3 = c0_2["logMh"][:]
Ms3 = c0_2["logMs"][:]
print(max(Mh3), max(Mh0))
Mh_filt = [10**c0_2["logMh"][:]>=minMh][0]

#keep_mask = SHMR_mask(Mh0, Ms0, Mh3, Ms3, sig = 2.5, n_bins = 10000, threshold = np.log10(7e+12))
keep_mask = SHMR_mask_high_bias(Mh0, Ms0, Mh3, Ms3, sig = 2.5, n_bins = 10000, threshold = np.log10(7e+12), bias_fact = 0.25)
filtered_Mh3 = Mh3[keep_mask]
filtered_Ms3 = Ms3[keep_mask]
len(filtered_Ms3)
#for i in range(len(m_bins)):
# %%
fig5, ax5 = plt.subplots(2,1, constrained_layout = True, sharex = True, sharey=True)
#ax5[0].hist2d(c0_2["logMh"][Mh_filt], c0_2["logMs"][Mh_filt], bins=100, density=True)
ax5[0].hist2d(filtered_Mh3, filtered_Ms3, bins=100, density=True)
#ax5[1].hist2d(np.log10(X1), np.log10(M1), bins=100, density=True)
ax5[1].hist2d(d0['logMh'], d0['logMs_or'], bins=100, density=True)
#P1=plt.scatter(X1, M1, s=1, color='red', zorder=10)
#P2=plt.scatter(X2, M2, s=1, color='blue', zorder=3)
#P3=plt.scatter(X3, M3, s=1, color='green')
ax5[0].set_ylim(np.log10(1e9), np.log10(4e12))
ax5[0].set_xlim(np.log10(minMh), np.log10(1e15))
#plt.yscale('log')
#plt.xscale('log')
# %%
plt.scatter(10**c0_2["logMh"][Mh_filt], 10**c0_2["logMs"][Mh_filt], s=0.1)
P1=plt.scatter(X1, M1, s=1, color='red', zorder=10)
P2=plt.scatter(X2, M2, s=1, color='blue', zorder=3)
#P3=plt.scatter(X3, M3, s=1, color='green')
plt.ylim(1e9, 2e13)
plt.xlim(2e10, 3e15)
plt.yscale('log')
plt.xscale('log')
# %%
model.get_config()
model.layers[29], model.layers[29].get_weights()

# %%
