#%%
from basemodule import *
from nnmodule import *
from NewFuncs import *
import h5py
import numpy as np
from Corrfunc.theory.wp import wp
from Corrfunc.theory.xi import xi
import matplotlib.pyplot as plt
import scienceplots
plt.style.use("science")
plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
fontT = 16
font = 14
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams['text.usetex'] = True

sim = 100
halo_filt = False

if sim == 100:
    d3_nn = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_obsdata_dark_100_model5_plus100.npz', 'r')
    model = keras.models.load_model('central-model5_plus100.h5')
    
    s3_nn = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_obsdata_dark_100_model5.npz', 'r')
    model2 = keras.models.load_model('satellite-model5.h5')
elif sim == 300:
    d3_nn = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_obsdata_dark_300.npz', 'r')
    model = keras.models.load_model('central-model14_plus.h5')
    
    s3_nn = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_obsdata_dark_300_plus300.npz', 'r')
    model2 = keras.models.load_model('satellite-model8.h5')

c0 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_train_plus.h5', 'r')
c1 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_test_plus.h5', 'r')
c3 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_dark.h5', 'r')

#hc23_c = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_HC23_data.npz', 'r')
#hc23_s = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_HC23_data.npz', 'r')

#hc23_M = np.concatenate((hc23_c["logMs"][:], hc23_s["logMs"][:]))

#Colour data
d0_c = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_obsdata_train.npz', 'r')
d1_c = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_obsdata_test.h5', 'r')
d3_c = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_obsdata_dark.h5', 'r')


s0 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_nndata_train_plus.h5', 'r')
s1 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_nndata_test_plus.h5', 'r')
s3 = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_nndata_dark.h5', 'r')

#Colour data
s0_c = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_obsdata_train.npz', 'r')
s1_c = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_obsdata_test.h5', 'r')
s3_c = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/satellite_obsdata_dark.h5', 'r')
d3_mh = np.load('/home/oscar239146/Uni/NN_Thesis/TNG-Networks-main/TNG100_Cent_logmh.npy')

time = c3['time'][:]
age = c3['lookback_time'][:]

sims0 = c0["simu"][:]
sims1 = c1["simu"][:]
sims3 = c3["simu"][:]
simss0 = s0["simu"][:]
simss1 = s1["simu"][:]
simss3 = s3["simu"][:]


#Change to reflect which sim wanted (Use 0 for both sims)
print("Sorting Data")
d0 = {key: c0[key] for key in c0.keys()}  
d1 = {key: c1[key] for key in c1.keys()} 
d3 = {key: c3[key] for key in c3.keys()}  
s0 = {key: s0[key] for key in s0.keys()}  
s1 = {key: s1[key] for key in s1.keys()} 
s3 = {key: s3[key] for key in s3.keys()}  

if sim != 0:
    print("Filtering Centrals ... ")
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
    for key in d3:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = d3[key]
            x = x[sims3 == sim]
            d3[key] = x
    print("Centrals Filtered. ")



#d0_c = {key: c0_c[key] for key in c0_c.keys()}  
#d1_c = {key: c1_c[key] for key in c1_c.keys()} 
#d3_c = {key: c3_c[key] for key in c3_c.keys()}  
#d3_nn = {key: c3_nn[key] for key in c3_nn.keys()} 
#s0_c = {key: s0_c[key] for key in s0_c.keys()}  
#s1_c = {key: s1_c[key] for key in s1_c.keys()} 
#s3_c = {key: s3_c[key] for key in s3_c.keys()} 
#s3_nn = {key: s3_nn[key] for key in s3_nn.keys()}  

"""
NN model colours already filtered by sim
"""
d0_gr = d0_c["magnitude"][:,1] - d0_c["magnitude"][:,2]
d1_gr = d1_c["mag_g"][:] - d1_c["mag_r"][:]
d3_gr = d3_c["mag_g"][:] - d3_c["mag_r"][:]
d3_nn_gr = d3_nn["magnitude"][:,1] - d3_nn["magnitude"][:,2]
s0_gr = s0_c["magnitude"][:,1] - s0_c["magnitude"][:,2]
s1_gr = s1_c["mag_g"][:] - s1_c["mag_r"][:]
s3_gr = s3_c["mag_g"][:] - s3_c["mag_r"][:]
s3_nn_gr = s3_nn["magnitude"][:,1] - s3_nn["magnitude"][:,2]

# Centrals
d0xt = np.stack((d0['Mhdot'][:], d0['delta1'][:], d0['delta3'][:], d0['delta5'][:], d0['vcirc'][:], d0['rhalf'][:], d0['skew'][:], d0['minD'][:]), axis=-1)
d0xs = np.stack((d0['beta'][:], d0['d_min'][:], d0['d_node'][:], d0['d_saddle_1'][:], d0['d_saddle_2'][:], d0['d_skel'][:], d0['formtime'][:], d0['logMh'][:], d0['logmaxMhdot'][:]), axis=-1)

d1xs = np.stack((d1['beta'][:], d1['d_min'][:], d1['d_node'][:], d1['d_saddle_1'][:], d1['d_saddle_2'][:], d1['d_skel'][:], d1['formtime'][:], d1['logMh'][:], d1['logmaxMhdot'][:]), axis=-1)

SFH0 = np.concatenate((d0['SFH'][:], d1['SFH'][:]), axis=0)
#SFH0 = d0['SFH'][:]
SFHq, SFHg, SFHo = GQTvecnorm(SFH0.reshape(*np.shape(SFH0), 1)) #added offsets to output
#SFH0 = s0['SFH'][:]
SFH0 = np.concatenate((s0['SFH'][:], s1['SFH'][:]), axis=0)
SFHq, SFHsg, SFHso = GQTvecnorm(SFH0.reshape(*np.shape(SFH0), 1)) #added offsets to output

ZH0 = np.concatenate((d0['ZH'][:], d1['ZH'][:]), axis=0)
#ZH0 = d0['ZH'][:]
ZHq, ZHg, ZHo = GQTvecnorm(ZH0.reshape(*np.shape(ZH0), 1))

others1 = np.concatenate((d1['logZ'][:].reshape(len(d1xs), 1), d1['logMs'][:].reshape(len(d1xs), 1),d1['mwsa'][:].reshape(len(d1xs), 1)), axis=-1)

others0 = np.concatenate((d0['logZ'][:].reshape(len(d0xs), 1), d0['logMs'][:].reshape(len(d0xs), 1), d0['mwsa'][:].reshape(len(d0xs), 1)), axis=-1)
others0 = np.vstack((others0,others1))
othq, othg, otho = GQTscalnorm(others0)

d3xt = np.stack((d3['Mhdot'][:], d3['delta1'][:], d3['delta3'][:], d3['delta5'][:], d3['vcirc'][:], d3['rhalf'][:], d3['skew'][:], d3['minD'][:]), axis=-1) #Temporal quantities (33 timesteps, 8 variables)
d3xs = np.stack((d3['beta'][:], d3['d_min'][:], d3['d_node'][:], d3['d_saddle_1'][:], d3['d_saddle_2'][:], d3['d_skel'][:], d3['formtime'][:], d3['logMh'][:], d3['logmaxMhdot'][:]), axis=-1) #Non-temporal quantities (9 variables)

d3xt1q, d3xt1g = GQTvecnorm(d3xt[:,:,0].reshape(len(d3xt), 33, 1))[:2] #output only first 2 and reshape
d3xt2q, d3xt2g = GQTscalnorm(d3xt[:,:,1:4])[:2]
d3xt3q, d3xt3g = GQTvecnorm(d3xt[:,:,4:])[:2]

d3xtq = np.concatenate((d3xt1q, d3xt2q, d3xt3q), axis=-1)
d3xsq, d3xsg, d3xso = GQTscalnorm(d3xs)
d3xq = [d3xtq, d3xsq]
d3yq = model.predict(d3xq)
#%%
#Satellites
others1 = np.concatenate((s1['logZ'][:].reshape(len(s1["simu"]), 1), s1['logMs'][:].reshape(len(s1["simu"]), 1), s1['mwsa'][:].reshape(len(s1["simu"]), 1)), axis=-1)
others0 = np.concatenate((s0['logZ'][:].reshape(len(s0["simu"]), 1), s0['logMs'][:].reshape(len(s0["simu"]), 1), s0['mwsa'][:].reshape(len(s0["simu"]), 1)), axis=-1)
others0 = np.vstack((others0,others1))
othq, othsg, othso = GQTscalnorm(others0)

s3xt = np.stack((s3['Mhdot'][:], s3['mhdot'][:], s3['delta'][:], s3['vcirc'][:], s3['rhalf'][:], s3['skew'][:], s3['minD'][:]), axis=-1) #Temporal quantities (33 timesteps, 8 variables)
s3xs = np.stack((s3['beta_halo'][:], s3['beta_sub'][:], s3['a_inf'][:], s3['a_max'][:], s3['logMu'][:], s3['logVrel'][:], s3['formtime'][:], s3['logMh'][:], s3['logmaxMhdot'][:], s3['logmh'][:], s3['logmaxmhdot'][:]), axis=-1) #Non-temporal quantities (9 variables)

s3xt1q, s3xt1g = GQTvecnorm(s3xt[:,:,0:2])[:2] #output only first 2 and reshape
s3xt2q, s3xt2g = GQTscalnorm(s3xt[:,:,2].reshape(len(s3xt), 33, 1))[:2]
s3xt3q, s3xt3g = GQTvecnorm(s3xt[:,:,3:])[:2]

s3xtq = np.concatenate((s3xt1q, s3xt2q, s3xt3q), axis=-1)
s3xsq, s3xsg, s3xso = GQTscalnorm(s3xs)
s3xq = [s3xtq, s3xsq]
s3yq = model2.predict(s3xq)

#%%
#print(np.shape(d1['SFH'][:]), np.shape(d1['ZH'][:]), np.shape(d1['logZ'][:].reshape(len(d1xs), 1)), np.shape(d1['logMs'][:].reshape(len(d1xs), 1)), np.shape(d1['mwsa'][:].reshape(len(d1xs), 1)))

#logMs0 = np.concatenate((d0['logMs'][:], d1['logMs'][:]), axis=0)
#print(logMs0.reshape(len(logMs0), 1))
#logMsq, logMsg, logMso = GQTscalnorm(logMs0.reshape(len(logMs0), 1)) #added offsets to output


SFH = np.squeeze(GQTvecinv(d3yq[:,0:33].reshape(len(d3xs), 33, 1), SFHg, SFHo)) #add offsets
ZH = np.squeeze(GQTvecinv(d3yq[:,33:66].reshape(len(d3xs), 33, 1), ZHg, ZHo))

others = np.squeeze(GQTscalinv(d3yq[:,66:], othg, otho))

if sim != 0:
    print("Filtering Satellites ... ")
    for key in s0:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = s0[key]
            x = x[simss0 == sim]
            s0[key] = x
    for key in s1:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = s1[key]
            x = x[simss1 == sim]
            s1[key] = x
    for key in s3:
        if ((key!="time") & (key!="lookback_time")&(key!="frequency")&(key!="redshift")):
            x = s3[key]
            x = x[simss3 == sim]
            s3[key] = x
    print("Satellites Filtered. ")

otherss = np.squeeze(GQTscalinv(s3yq[:,66:], othsg, othso))
otherss = otherss[simss3 == sim]

d0_gr = d0_gr[sims0 == sim]
d1_gr = d1_gr[sims1 == sim]
d3_gr = d3_gr[sims3 == sim]

s0_gr = s0_gr[simss0 == sim]
s1_gr = s1_gr[simss1 == sim]
s3_gr = s3_gr[simss3 == sim]

#hc23_M = hc23_M[np.concatenate((sims3==sim, simss3==sim))]
#%%%
X1 = 10**d1['logMh'][:]
#X2 = 10**d2['logMh'][:]
X3 = 10**d3['logMh'][:]

#Z3 = d3yq[:,66] #predicted metallicity
#mass3 = 10**GQTscalinv(d3yq[:,67].reshape(len(d3yq), 1), logMsg, logMso) #predicted logMs

M1 = d1['logMS'][:]
#M2 = d2['logMS'][:]
M3 = sfhint(SFH, time=time)

Z1 = d1['logZS'][:]
#Z2 = d2['logZS'][:]
#Z3 = mwz(ZH, SFH, time=time)

A1 = mwa(age, d1['SFH'][:])

A3 = mwa(age, SFH)

M1 = 10**M1#Convert to non logged values
Z1 = 10**Z1


#%%
h = 0.7

nthreads = 2
pimax = 80/h #Bose2019
pimax100 = 20
nbins = 10
bins = np.logspace(-3,2.1,nbins+1) # Note the + 1 to nbins
bins100 = np.logspace(np.log10(0.1),np.log10(50),nbins+1)
#%%
def gr_split(logM_star):
    return (logM_star+np.log10(h))*0.054+0.05
    #return np.log10(M_star*h)*0.054 + 0.05

filtc = [d3["logMh"]<15][0]
#mass3 = np.concatenate((others[:,1][filtc], otherss[:,1])) #logMs
mass3 = np.concatenate((others[:,1], otherss[:,1])) #logMs
Z3 = np.concatenate((others[:,0], otherss[:,0]))
mwa3 = np.concatenate((others[:,2], otherss[:,2]))



c_gr0 = np.concatenate((d0_gr, s0_gr))
c_gr1 = np.concatenate((d1_gr, s1_gr))
c_gr00 = np.concatenate((c_gr0, c_gr1))

#mass100 = mass[sims == 100]
#c_mass3 = np.concatenate((d3["logMs"][filtc], s3["logMs"][:]))
c_mass3 = np.concatenate((d3["logMs"][:], s3["logMs"][:]))
#c_mass3 = s3["logMs"][:]
c_gr3 = np.concatenate((d3_gr, s3_gr))
c_nn_gr3 = np.concatenate((d3_nn_gr, s3_nn_gr))
c_Z3 = np.concatenate((d3["logZ"][:], s3["logZ"][:]))
c_mwa3 = np.concatenate((d3["mwsa"][:], s3["mwsa"][:]))

#mass3 = c_mass
#s_mass = s1["logMs"][:]
#cmass300 = c_mass[c_sims == 300]
#cmass100 = c_mass[c_sims == 100]
#smass300 = s_mass[s_sims == 300]
#smass100 = s_mass[s_sims == 100]


#coms = np.concatenate((d3["centreofmass"][:,32,:][filtc], s3["centreofmass"][:,32,:]))
coms = np.concatenate((d3["centreofmass"][:,32,:], s3["centreofmass"][:,32,:]))
#coms = s3["centreofmass"][:,32,:]

#filtc0 = [d0["logMh"]>=min(d3["logMh"])][0]
#filts0 = [s0["logmh"]>=min(s3["logmh"])][0]
#filtc1 = [d1["logMh"]>=min(d3["logMh"])][0]
#filts1 = [s1["logmh"]>=min(s3["logmh"])][0]

filtc0 = [d0["logMh"]>=11][0]
filts0 = [s0["logMh"]>=11][0]
filtc1 = [d1["logMh"]>=11][0]
filts1 = [s1["logMh"]>=11][0]
filtc3 = [d3["logMh"]>=11][0]
filts3 = [s3["logMh"]>=11][0]

#coms0 = np.concatenate((d0["centreofmass"][:,32,:][filtc0], s0["centreofmass"][:,32,:][filts0]))
coms0 = np.concatenate((d0["centreofmass"][:,32,:][:], s0["centreofmass"][:,32,:]))

#coms1 = np.concatenate((d1["centreofmass"][:,32,:][filtc1], s1["centreofmass"][:,32,:]))
coms1 = np.concatenate((d1["centreofmass"][:,32,:][:], s1["centreofmass"][:,32,:]))

if halo_filt:
    coms0 = coms0[np.concatenate((filtc0, filts0))]
    coms1 = coms1[np.concatenate((filtc1, filts1))]
    c_gr00 = c_gr00[np.concatenate((filtc0, filts0, filtc1, filts1))]
    coms = coms[np.concatenate((filtc3, filts3))]
    mass3 = mass3[np.concatenate((filtc3, filts3))]
    c_mass3 = c_mass3[np.concatenate((filtc3, filts3))]
    c_gr3 = c_gr3[np.concatenate((filtc3, filts3))]
    c_nn_gr3 = c_nn_gr3[np.concatenate((filtc3, filts3))]
    c_Z3 = c_Z3[np.concatenate((filtc3, filts3))]
    Z3 = Z3[np.concatenate((filtc3, filts3))]
    mwa3 = mwa3[np.concatenate((filtc3, filts3))]
    c_mwa3 = c_mwa3[np.concatenate((filtc3, filts3))]
    
    
coms00 = np.concatenate((coms0, coms1))
#%%
Mh3 = np.concatenate((d3["logMh"], s3["logmh"]))#[np.concatenate((filtc3, filts3))]

#%%

rng = np.random.default_rng(46)

#%%
#Colour split

if sim == 100:
    n_bins = 4
    m_bins = np.linspace(8.5, 11, n_bins + 1)
    m_bins0 = m_bins
    mass0 = np.concatenate((d0["logMs"][:], s0["logMs"][:])) #_or suffix if no res-correct
    Z0 = np.concatenate((d0["logZ"], s0["logZ"]))
    mwa0 = np.concatenate((d0["mwsa"], s0["mwsa"]))
    mass1 = np.concatenate((d1["logMs"][:], s1["logMs"][:]))
    Z1 = np.concatenate((d1["logZ"], s1["logZ"]))
    mwa1 = np.concatenate((d1["mwsa"], s1["mwsa"]))

    mass00 = np.concatenate((mass0, mass1))
    Z00 = np.concatenate((Z0, Z1))
    mwa00 = np.concatenate((mwa0, mwa1))
    
    if halo_filt:
        mass00 = mass00[np.concatenate((filtc0, filts0,filtc1, filts1))]
        Z00 = Z00[np.concatenate((filtc0, filts0,filtc1, filts1))]
        mwa00 = mwa00[np.concatenate((filtc0, filts0,filtc1, filts1))]

    
print(np.percentile(mass3, 1),np.percentile(mass3, 99))
print(np.percentile(mass1, 10),np.percentile(mass1, 90))

#%%
chi2_nn = []
chi2_hc24 = []

volbin_num = 2
fig3, ax4 = plt.subplots(2,2,  figsize = (10,6), sharey=True,sharex=True, constrained_layout = True)
fig5, ax5 = plt.subplots(2,2,  figsize = (10,6), sharey=True, sharex=True, constrained_layout = True)
fig3.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, n_bins))))
mwa_gate = False
Z_gate = False
if sim == 100:
    for i in range(0,n_bins):
        blue_cut = gr_split((m_bins[i+1]+m_bins[i])/2)
        ax4[i//2,i%2].set_xscale("log")
        ax4[i//2,i%2].set_xlim(0.15, 30)
        ax4[i//2,i%2].set_ylim(1, 1e+4)
        ax4[i//2,i%2].set_yscale("log")    
        ax4[i//2,i%2].set_title(str(m_bins[i])+" - "+str(m_bins[i+1])+r" $\log{M_\star/M_\odot}$", fontsize=font-1)
        #ax4[i//2,i%2].set_xlim(0.03,60)
        
        wp_300_mass_b00 = wp(111, pimax100, 1, bins100, coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00<=blue_cut)], coms00[:,1][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00<=blue_cut)], coms00[:,2][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00<=blue_cut)], output_rpavg = True)
        ax4[i//2,i%2].errorbar(wp_300_mass_b00["rpavg"], wp_300_mass_b00["wp"], linestyle=":", color="blue",zorder=12)
        wp_b00 = wp_300_mass_b00["wp"]
        
        wp_300_mass_r00 = wp(111, pimax100, 1, bins100, coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00>blue_cut)], coms00[:,1][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00>blue_cut)], coms00[:,2][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00>blue_cut)], output_rpavg = True)
        ax4[i//2,i%2].errorbar(wp_300_mass_r00["rpavg"], wp_300_mass_r00["wp"], linestyle=":", color="red",zorder=12)
        wp_r00 = wp_300_mass_r00["wp"]
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], coms[:,2][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], volbin_num)
        ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], color="blue", yerr=wp_err*2)
        cov = JK_wp_cov(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], volbin_num)
        #print(cov)
        chi2_b = chisq_cov(wp_300_mass["wp"], wp_b00, cov, normalised = True)
        print(chi2_b)
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], coms[:,2][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], volbin_num)
        ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], color="red", yerr=wp_err*2)
        
        cov = JK_wp_cov(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], volbin_num)
        
        chi2_r = chisq_cov(wp_300_mass["wp"], wp_r00, cov, normalised = True)
        chi2_nn.append([chi2_b, chi2_r])
        
        print(len(coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])]))
    
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], coms[:,2][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], output_rpavg = True)
        
        
        if len(coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)]) > 1:
            wp_err = JK_wp_err(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], volbin_num)
            cov = JK_wp_cov(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], volbin_num)
            chi2_r = chisq_cov(wp_300_mass["wp"], wp_r00, cov, normalised = True)
            ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], linestyle="--",color="red", yerr=wp_err*2) 
        else:
            chi2_r = np.nan
            ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], linestyle="--",color="red") 

            
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], coms[:,2][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], output_rpavg = True)
        
        if len(coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)]) > 1:
            wp_err = JK_wp_err(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], volbin_num)
            cov = JK_wp_cov(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], volbin_num)
            chi2_b = chisq_cov(wp_300_mass["wp"], wp_b00, cov, normalised = True)
            ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], linestyle="--", color="blue", yerr=wp_err*2)
        else:
            chi2_b = np.nan
            ax4[i//2,i%2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], linestyle="--", color="blue")
        
        chi2_hc24.append([chi2_b, chi2_r])
        
        if mwa_gate:
            ax5[i//2,i%2].hist(mwa00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00>blue_cut)], color = "red", histtype = "step", density=True, linestyle=":")
            ax5[i//2,i%2].hist(mwa3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], color = "red", histtype = "step", density=True)
            ax5[i//2,i%2].hist(mwa00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00<=blue_cut)], color = "blue", histtype = "step", density=True, linestyle=":")
            ax5[i//2,i%2].hist(mwa3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], color = "blue", histtype = "step", density=True)
            ax5[i//2,i%2].hist(c_mwa3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], color = "blue", histtype = "step", density=True, linestyle="--")
            ax5[i//2,i%2].hist(c_mwa3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], color = "red", histtype = "step", density=True, linestyle="--")
        
        elif Z_gate:
            ax5[i//2,i%2].hist(Z00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00>blue_cut)], color = "red", histtype = "step", density=True, linestyle=":")
            ax5[i//2,i%2].hist(Z3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)], color = "red", histtype = "step", density=True)
            ax5[i//2,i%2].hist(Z00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])&(c_gr00<=blue_cut)], color = "blue", histtype = "step", density=True, linestyle=":")
            ax5[i//2,i%2].hist(Z3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)], color = "blue", histtype = "step", density=True)
            ax5[i//2,i%2].hist(c_Z3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut)], color = "blue", histtype = "step", density=True, linestyle="--")
            ax5[i//2,i%2].hist(c_Z3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut)], color = "red", histtype = "step", density=True, linestyle="--")
            ax5[i//2,i%2].set_yscale("log")
            ax5[i//2,i%2].set_xlabel(r"$\log{Z/Z_\odot}$") 
        
       
handles = [
            plt.Line2D([0], [0], linestyle='-', color='black', label="Modified NN"),
            plt.Line2D([0], [0], linestyle='--', color='black', label="HC24"),
            plt.Line2D([0], [0], linestyle=':', color='black', label="TNG-Bright")]
ax4[0,0].legend(handles = handles)
        

fig3.suptitle(r"Separated Blue-Red Galaxy Clustering, z = 0, TNG"+str(sim), fontsize=font)


fig3.supxlabel(r"$r_p$ [Mpc]", fontsize = font)
fig3.supylabel(r"$w_p (r_p)$", fontsize = font)

chi2_nn, chi2_hc24 = np.array(chi2_nn), np.array(chi2_hc24)

#%%
print("NN - Blue - Red")
print(chi2_nn)
print("Blue, Red Means:",np.mean(chi2_nn[:,0]), np.mean(chi2_nn[:,1]))
print("Blue, Red Medians:",np.median(chi2_nn[:,0]), np.median(chi2_nn[:,1]))

print("HC24 - Blue - Red")
print(chi2_hc24)
print("Blue, Red Means:",np.mean(chi2_hc24[1,0]), np.mean(chi2_hc24[1,1]))
print("Blue, Red Medians:",np.median(chi2_hc24[1,0]), np.median(chi2_hc24[1,1]))

#%%
#XY Distribution
n_bins = 4
#m_bins = np.linspace(8.5, 11.5, n_bins + 1)
m_bins = np.linspace(9, 11, n_bins + 1)
print(m_bins)
Mh0 = np.concatenate((d0["logMh"], s0["logMh"], d1["logMh"], s1["logMh"]))
minMh = np.min(np.concatenate((d3["logMh"], s3["logMh"])))
Mh00 = Mh0
fig3, ax4 = plt.subplots(1,3,  figsize = (12,6), sharey=True, constrained_layout = True)
fig3.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, n_bins))))
if sim == 100:
    for i in range(0,n_bins):
        
        #ax4[i//2,i%2].set_title(str(m_bins[i])+" - "+str(m_bins[i+1])+r" $\log{M_\odot}$")
        
        
        blue_cut = gr_split((m_bins[i+1]+m_bins[i])/2)
        ax4[0].scatter(coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut) & (coms[:,2]<30)], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3<=blue_cut)& (coms[:,2]<30)], color="blue", s=0.1)

        
        ax4[0].scatter(coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)& (coms[:,2]<30) & (coms[:,2]<30)], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])&(c_nn_gr3>blue_cut)& (coms[:,2]<30) & (coms[:,2]<30)], color="red", s=0.1)


        ax4[1].scatter(coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut) & (coms[:,2]<30)], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3>blue_cut) & (coms[:,2]<30)], color="red", s=0.1)
        
        ax4[1].scatter(coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut) & (coms[:,2]<30)], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])&(c_gr3<=blue_cut) & (coms[:,2]<30)], color="blue", s=0.1)
        
        
        ax4[2].scatter(coms00[:,0][(mass00>=m_bins[i]) & (mass00<m_bins[i+1])&(c_gr00<=blue_cut) & (coms00[:,2]<30)], coms00[:,1][(mass00>=m_bins[i]) & (mass00<m_bins[i+1])&(c_gr00<=blue_cut) & (coms00[:,2]<30)], color="blue", s=0.1)
        
        ax4[2].scatter(coms00[:,0][(mass00>=m_bins[i]) & (mass00<m_bins[i+1])&(c_gr00>blue_cut) & (coms00[:,2]<30)], coms00[:,1][(mass00>=m_bins[i]) & (mass00<m_bins[i+1])&(c_gr00>blue_cut) & (coms00[:,2]<30)], color="red", s=0.1)
print(np.sum([(Mh0>minMh)]))
ax4[0].set_title("Refined NN Dark Predictions")
ax4[1].set_title("HC24 Dark Predictions")
ax4[2].set_title("Bright Data")


# %%
import matplotlib.gridspec as gridspec
colour_bins = np.linspace(np.min(c_gr00), np.max(c_gr00), 50)
fig = plt.figure(constrained_layout=True, figsize=(12, 4))
gs = gridspec.GridSpec(1, 9, figure=fig) 


ax_scatter = [fig.add_subplot(gs[0, i*3:i*3+2]) for i in range(3)]
ax_hist = [fig.add_subplot(gs[0, i*3+2], sharey=ax_scatter[i]) for i in range(3)]

ax_scatter[0].scatter(mass3, c_nn_gr3, s=0.1, color="green")
ax_scatter[1].scatter(mass00, c_gr00, s=0.1, color="blue")
ax_scatter[2].scatter(c_mass3, c_gr3, s=0.1, color="red")


ax_hist[0].hist(c_nn_gr3, bins=colour_bins, orientation='horizontal', color="green", alpha=0.6, density = True)
ax_hist[1].hist(c_gr00, bins=colour_bins, orientation='horizontal', color="blue", alpha=0.6, density = True)
ax_hist[2].hist(c_gr3, bins=colour_bins, orientation='horizontal', color="red", alpha=0.6, density = True)

for ax in ax_scatter:
    ax.set_xlim(9,12.1)
    ax.set_ylim(0.1,1.2)
    ax.set_xlabel(r"$\log{M_\star / M_\odot}$", fontsize=12+3)
ax_scatter[1].tick_params(labelleft=False)
ax_scatter[2].tick_params(labelleft=False)

for ax in ax_hist:
    ax.tick_params(labelleft=False)  # hide ticks
    ax.set_xlabel("Prob. Density", fontsize=12+3)

fig.supylabel(r"$g-r$ Mag. Colour", fontsize=12+3)
#fig.supxlabel(r"$\log{M_\star / M_\odot}$")
fig.suptitle("TNG100 Colour - Stellar Mass Diagrams", fontsize=13+3)

ax_scatter[0].set_title("Modified NN (Dark Predictions)", fontsize=12+3)
ax_scatter[1].set_title("TNG-Bright Data", fontsize=12+3)
ax_scatter[2].set_title("HC24 (Dark Predictions)", fontsize=12+3)

# %%
#print(grads00)
print(grads3_err)
print(grads00_err)
print(grads_hc_err)
graderr0 = np.concatenate((grads00, grads00_err), axis=1)
graderr3 = np.concatenate((grads3, grads3_err), axis=1)
graderr_hc = np.concatenate((grads_hc, grads_hc_err), axis=1)
#print(graderr3)
#np.savetxt("TNG100_grads.txt", np.vstack((graderr0, graderr3, graderr_hc)))

# %%
#Normal wp
mwa_gate = False
Z_gate = False
if sim == 100:
    n_bins = 3
    m_bins = np.linspace(8.5, 11.5, n_bins + 1)
    m_bins0 = m_bins

if mwa_gate or Z_gate:
    fig5, ax5 = plt.subplots(2,2,  figsize = (10,6), sharey=True, sharex=True, constrained_layout = True)
    
fig3, ax4 = plt.subplots(1,3,  figsize = (12,6), sharey=True, constrained_layout = True)
grads3 = []
grads3_err = []
grads00 = []
grads00_err = []
grads_hc = []
grads_hc_err = []
#fig3.gca().set_prop_cycle(plt.cycler('color', plt.cm.tab10(np.linspace(0, 1, n_bins))))
volbin_num300 = 2
volbin_num = 2
if sim == 100:
    for i in range(0,n_bins):
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,2][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], volbin_num)
        ax4[0].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], yerr=wp_err*2, label = str(m_bins[i])+"-"+str(m_bins[i+1]))
        rise1 = np.log10(wp_300_mass["wp"][3])-np.log10(wp_300_mass["wp"][0])
        run1 = np.log10(wp_300_mass["rpavg"][3])-np.log10(wp_300_mass["rpavg"][0])
        gamma1 = rise1/run1
        
        err1 = abs(1/run1) * np.sqrt((wp_err[0]/(wp_300_mass["rpavg"][0]*2.3026))**2 + (wp_err[3]/(wp_300_mass["rpavg"][3]*2.3026))**2)
        #err1 = gamma1 * np.sqrt(wp_err[0]**2 + wp_err[3]**2)
        
        rise2 = np.log10(wp_300_mass["wp"][-2])-np.log10(wp_300_mass["wp"][4])
        run2 = np.log10(wp_300_mass["rpavg"][-2])-np.log10(wp_300_mass["rpavg"][4])
        gamma2 = rise2/run2
        grads3.append([gamma1, gamma2])
        err2 = abs(1/run2) * np.sqrt((wp_err[4]/(wp_300_mass["rpavg"][4]*2.3026))**2 + (wp_err[-2]/(wp_300_mass["rpavg"][-2]*2.3026))**2)
        grads3_err.append([err1, err2])
        
        #Use 2-sigma CI
    print(grads3)
    for i in range(0,n_bins):
        wp_300_mass = wp(111, pimax100, 1, bins100, coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,1][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,2][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], volbin_num)
        ax4[1].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], label = str(m_bins0[i])+"-"+str(m_bins0[i+1]), yerr=wp_err)
        rise1 = np.log10(wp_300_mass["wp"][3])-np.log10(wp_300_mass["wp"][0])
        run1 = np.log10(wp_300_mass["rpavg"][3])-np.log10(wp_300_mass["rpavg"][0])
        gamma1 = rise1/run1
        err1 = abs(1/run1) * np.sqrt((wp_err[0]/(wp_300_mass["rpavg"][0]*2.3026))**2 + (wp_err[3]/(wp_300_mass["rpavg"][3]*2.3026))**2)
        
        rise2 = np.log10(wp_300_mass["wp"][-2])-np.log10(wp_300_mass["wp"][4])
        run2 = np.log10(wp_300_mass["rpavg"][-2])-np.log10(wp_300_mass["rpavg"][4])
        gamma2 = rise2/run2
        grads00.append([gamma1, gamma2])
        err2 = abs(1/run2) * np.sqrt((wp_err[4]/(wp_300_mass["rpavg"][4]*2.3026))**2 + (wp_err[-2]/(wp_300_mass["rpavg"][-2]*2.3026))**2)
        grads00_err.append([err1, err2])
        
        
    for i in range(0,n_bins):
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], coms[:,2][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], output_rpavg = True)
        if len(coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])])>0:
            wp_err = JK_wp_err(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], volbin_num)
            ax4[2].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], label = str(m_bins[i])+"-"+str(m_bins[i+1]), yerr=wp_err*2)
        rise1 = np.log10(wp_300_mass["wp"][3])-np.log10(wp_300_mass["wp"][0])
        run1 = np.log10(wp_300_mass["rpavg"][3])-np.log10(wp_300_mass["rpavg"][0])
        gamma1 = rise1/run1
        err1 = abs(1/run1) * np.sqrt((wp_err[0]/(wp_300_mass["rpavg"][0]*2.3026))**2 + (wp_err[3]/(wp_300_mass["rpavg"][3]*2.3026))**2)
        
        rise2 = np.log10(wp_300_mass["wp"][-2])-np.log10(wp_300_mass["wp"][4])
        run2 = np.log10(wp_300_mass["rpavg"][-2])-np.log10(wp_300_mass["rpavg"][4])
        gamma2 = rise2/run2
        grads_hc.append([gamma1, gamma2])
        err2 = abs(1/run2) * np.sqrt((wp_err[4]/(wp_300_mass["rpavg"][4]*2.3026))**2 + (wp_err[-2]/(wp_300_mass["rpavg"][-2]*2.3026))**2)
        grads_hc_err.append([err1, err2])

for i in range(0,n_bins):
        j = i
        if mwa_gate:
            ax5[j//2,j%2].hist(mwa00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], color = "red", histtype = "step", density=True, linestyle=":")
            ax5[j//2,j%2].hist(mwa3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], color = "green", histtype = "step", density=True)
            ax5[j//2,j%2].hist(c_mwa3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], color = "blue", histtype = "step", density=True, linestyle="--")
        
        elif Z_gate:
            ax5[j//2,j%2].hist(Z00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], color = "blue", histtype = "step", density=True, linestyle=":")
            ax5[j//2,j%2].hist(Z3[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], color = "green", histtype = "step", density=True)
            ax5[j//2,j%2].hist(c_Z3[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], color = "red", histtype = "step", density=True, linestyle="--")

            ax5[j//2,j%2].set_yscale("log")
            ax5[j//2,j%2].set_xlabel(r"$\log{Z/Z_\odot}$") 
        
ax4[0].legend(title=r"$\log{M_\star/M_\odot}$")
ax4[0].set_xscale("log")
ax4[0].set_yscale("log")
ax4[1].set_xscale("log")
ax4[1].set_yscale("log")
ax4[2].set_xscale("log")
ax4[2].set_yscale("log")
#ax4[1].legend()
#ax4[2].legend()
fig3.suptitle(r"Both Galaxies, z = 0, TNG"+str(sim), fontsize=14)
ax4[0].set_title(r"Modified NN (Dark Predictions)", fontsize = font)
ax4[1].set_title(r"TNG-Bright Data", fontsize = font)
ax4[2].set_title(r"HC24 NN (Dark Predictions)", fontsize = font)

if sim == 100:
    ax4[0].set_xlim(0.1,30)
    ax4[1].set_xlim(0.1,30)
    ax4[2].set_xlim(0.1,30)
    ax4[0].set_ylim(0.9,2e+3)
    
fig3.supxlabel(r"$r_p$ [Mpc]", fontsize = font)
fig3.supylabel(r"$w_p (r_p)$", fontsize = font)

#%%
#DM halo wp
if sim == 100:
    n_bins = 3
    m_bins = np.linspace(11, 14, n_bins + 1)
    #m_bins = np.linspace(9.5, 13, n_bins + 1)
    m_bins0 = m_bins
elif sim == 300:
    n_bins = 4
    #m_bins = np.linspace(10, 11.5, n_bins + 1)
    m_bins = np.linspace(9.5, 14.5, n_bins + 1)
    m_bins = np.linspace(12, 15, n_bins + 1) #approx 10-90 dark percentiles
    #m_bins = [10, 11, 11.5, 11.7, 12]
    #m_bins = [9, 10.5, 11.25, 11.5, 12] #Best mass bins
    #m_bins = [9, 10, 10.5, 11, 12]
    m_bins0 = m_bins#np.linspace(9, 11, n_bins + 1)

Mh3 = np.concatenate((d3_mh, s3["logMh"]))
#Mh3 = np.concatenate((d3["logMh"], s3["logMh"]))#[np.concatenate((filtc3, filts3))]
Mh00 = np.concatenate((d0["logMh"],s0["logMh"],d1["logMh"],s1["logMh"]))#[np.concatenate((filtc0, filts0, filtc1, filts1))]
#font = font+5
fig3, ax4 = plt.subplots(1,2,  figsize = (6,4), sharey=True, sharex=True,constrained_layout = True)
#fig3.gca().set_prop_cycle(plt.cycler('color', plt.cm.tab10(np.linspace(0, 1, n_bins))))
volbin_num300 = 2
volbin_num = 2
if sim == 100:
    for i in range(0,n_bins):
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])], coms[:,1][(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])], coms[:,2][(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms[(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])], volbin_num)
        ax4[0].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], yerr=wp_err*2, label = "{0:3.1f}-{1:3.1f}".format(m_bins[i], m_bins[i+1]))
        #Use 2-sigma CI
    
    for i in range(0,n_bins):
        wp_300_mass = wp(111, pimax100, 1, bins100, coms00[:,0][(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1])], coms00[:,1][(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1])], coms00[:,2][(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms00[(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1])], volbin_num)
        ax4[1].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], label = "{0:3.1f}-{1:3.1f}".format(m_bins[i], m_bins[i+1]), yerr=wp_err)
        

        
ax4[0].legend(title = r"$\log{M_H / M_\odot}$", loc="lower left", title_fontsize = font-6, fontsize=font-6)
ax4[0].set_xscale("log")
ax4[0].set_yscale("log")
ax4[1].set_xscale("log")
ax4[1].set_yscale("log")

#ax4[1].legend()

fig3.suptitle(r"Halo 2-Point Correlation Function, z = 0, TNG"+str(sim), fontsize=font)
ax4[0].set_title(r"TNG-Dark Halos", fontsize=font)
ax4[1].set_title(r"TNG-Bright Halos", fontsize=font)

ax4[0].set_xlim(0.09,30)
ax4[0].set_ylim(0.2,1e+4)

fig3.supxlabel(r"$r_p$ [Mpc]", fontsize = font)
fig3.supylabel(r"$w_p (r_p)$", fontsize = font)
#%%
falses = []
for i in range(len(Mh00)):
    if (i<len(d0["logMh"])) or ((i>=len(d0["logMh"])+len(s0["logmh"])) and (i<len(d0["logMh"])+len(s0["logmh"])+len(d1["logMh"]))):
        falses.append(False)
    else:
        falses.append(True)
falses = np.array(falses)

falses2 = []
for i in range(len(Mh3)):
    if (i<len(d3["logMh"])):
        falses2.append(False)
    else:
        falses2.append(True)
falses2 = np.array(falses2)

#%%
#DM halo XY Distribution
#n_bins = 6
#m_bins = np.linspace(8.5, 11.5, n_bins + 1)
#m_bins = np.linspace(8.5, 11.5, n_bins + 1)
print(m_bins)
cmap = plt.get_cmap("viridis")
false_it=False
colours = cmap(np.linspace(0, 0.5,n_bins))
fig3, ax4 = plt.subplots(n_bins,2,  figsize = (7,8), sharey=True, constrained_layout = True, sharex=True)
false_it = False
#colours = ["blue", "green", "yellow", "orange", "red", "black"]
print(colours)
if sim == 100:
    for i in range(0,n_bins):
        
        #ax4[i//2,i%2].set_title(str(m_bins[i])+" - "+str(m_bins[i+1])+r" $\log{M_\odot}$")
        if false_it:
            ax4[i,0].scatter(coms[falses2][:,0][(Mh3[falses2]>=m_bins[i]) & (Mh3[falses2]<m_bins[i+1])& (coms[falses2][:,2]<25)], coms[falses2][:,1][(Mh3[falses2]>=m_bins[i]) & (Mh3[falses2]<m_bins[i+1])& (coms[falses2][:,2]<25)], label = "{0:3.1f}-{1:3.1f}".format(m_bins[i], m_bins[i+1]), s=0.2, color = colours[i])
        else:
            ax4[i,0].scatter(coms[:,0][(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])& (coms[:,2]<25)], coms[:,1][(Mh3>=m_bins[i]) & (Mh3<m_bins[i+1])& (coms[:,2]<25)], label = "{0:3.1f}-{1:3.1f}".format(m_bins[i], m_bins[i+1]), s=0.2, color = colours[i])
        
        if false_it:
            ax4[i,1].scatter(coms00[falses][:,0][(Mh00[falses]>=m_bins0[i]) & (Mh00[falses]<m_bins0[i+1])& (coms00[falses][:,2]<25)], coms00[falses][:,1][(Mh00[falses]>=m_bins0[i]) & (Mh00[falses]<m_bins0[i+1]) & (coms00[falses][:,2]<25)], label=m_bins[i], s=0.2, color = colours[i])
        else:
            ax4[i,1].scatter(coms00[:,0][(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1])& (coms00[:,2]<25)], coms00[:,1][(Mh00>=m_bins0[i]) & (Mh00<m_bins0[i+1]) & (coms00[:,2]<25)], label=m_bins[i], s=0.2, color = colours[i])
        ax4[i,0].legend(frameon=True, title=r"$\log{M_\star/M_\odot}$", loc="upper right")

        
#print(np.sum([(Mh0>minMh)]))
ax4[0,0].set_title("Dark Subhalos")
ax4[0,1].set_title("Bright Subhalos")
#ax4[0].legend()
#ax4[1].legend()

# %%
#plt.hist(np.concatenate((d0["logMh"], s0["logmh"])), histtype = "step", density=True, color = "blue")
#plt.hist(Mh3, histtype = "step", density=True, color = "green")
#plt.hist(np.concatenate((d0["logMh"], s0["logmh"])), histtype = "step", density=True)
# %%

#plt.hist(mass3, density = True, histtype = "step", color = "green")
#plt.hist(mass00, density = True, histtype = "step", color = "blue")
# %%
#plt.hist(otherss[:,1], density = True, histtype = "step", color = "green")
#plt.hist(s0["logMs_or"], density = True, histtype = "step", color = "blue")
#plt.hist(others[:,1], density = True, histtype = "step", color = "green")
#plt.hist(d0["logMs_or"], density = True, histtype = "step", color = "blue")

#model.get_config()
#%%
if sim == 100:
    n_bins = 6
    m_bins = np.linspace(9, 11.5, n_bins + 1)
    m_bins0 = m_bins
elif sim == 300:
    n_bins = 6*2
    #m_bins = np.linspace(10, 11.5, n_bins + 1)
    m_bins = np.linspace(9, 12, n_bins + 1)
    #m_bins = np.linspace(10, 12, n_bins + 1) #approx 10-90 dark percentiles
    #m_bins = [10, 11, 11.5, 11.7, 12]
    #m_bins = [9, 10.5, 11.25, 11.5, 12] #Best mass bins
    #m_bins = [9, 10, 10.5, 11, 12]
    m_bins0 = m_bins#np.linspace(9, 11, n_bins + 1)

fig3, ax4 = plt.subplots(1,1, constrained_layout = True)
#fig3.gca().set_prop_cycle(plt.cycler('color', plt.cm.tab10(np.linspace(0, 1, n_bins))))
volbin_num300 = 2
volbin_num = 2
if sim == 100:
    for i in range(0,n_bins):
        
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,2][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms[(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], volbin_num)
        ax4.errorbar(m_bins[i], max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)]), yerr=wp_err[(wp_300_mass["wp"]==max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)])) & (wp_300_mass["rpavg"]>=0.03)]*2, label = "Modified NN", color = "green", marker = ".", zorder=12)
        #Use 2-sigma CI
    
    for i in range(0,n_bins):
        wp_300_mass = wp(111, pimax100, 1, bins100, coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,1][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,2][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], output_rpavg = True)
        wp_err = JK_wp_err(111, 20, 1, bins100, coms00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], volbin_num)
        ax4.errorbar(m_bins[i], max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)]), yerr=wp_err[(wp_300_mass["wp"]==max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)])) & (wp_300_mass["rpavg"]>=0.03)]*2, label = "Bright Data", color = "blue", marker = ".", zorder=11)
        
        
    for i in range(0,n_bins):
        wp_300_mass = wp(111, 20, 1, bins100, coms[:,0][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], coms[:,1][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], coms[:,2][(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], output_rpavg = True)
        if len(coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])]>1):
            wp_err = JK_wp_err(111, 20, 1, bins100, coms[(c_mass3>=m_bins[i]) & (c_mass3<m_bins[i+1])], volbin_num300)
            ax4.errorbar(m_bins[i], max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)]), yerr=wp_err[(wp_300_mass["wp"]==max(wp_300_mass["wp"][(wp_300_mass["rpavg"]>=0.03)])) & (wp_300_mass["rpavg"]>=0.03)]*2, label = "HC24", color = "red", marker = ".", zorder=8)

    
        
#ax4.legend()
#ax4[0].set_xscale("log")
ax4.set_yscale("log")

fig3.suptitle(r"Both Galaxies, z = 0, $r_p > 0.03$ Mpc, TNG"+str(sim), fontsize=14)

ax4.set_xlim(min(m_bins)-0.5,max(m_bins))
#ax4[1].set_xlim(0.03,60)
#ax4[2].set_xlim(0.03,60)
fig3.supxlabel(r"$\log {M_{\star} / M_{\odot}}$", fontsize = font)
fig3.supylabel(r"$w_p$ Amplitude [Mpc]", fontsize = font)


# %%
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline

if sim == 100:
    n_bins = 3
    m_bins = np.linspace(9, 11.5, n_bins + 1)
    m_bins0 = m_bins
elif sim == 300:
    n_bins = 6

    m_bins = np.linspace(8.5, 11.5, n_bins + 1)

    m_bins0 = m_bins#np.linspace(9, 11, n_bins + 1)
rp_vals = np.logspace(-1, 2, 100)
rp_vals2 = np.logspace(-1, 2, 1000)
fig5, ax5 = plt.subplots(1,2,  figsize = (12,6),  constrained_layout = True)
colours = ["indigo", "blue", "red", "green", "gold", "purple"]
if sim != 100:
    print("DARK")
    for i in range(1,n_bins):
        
        wp_300_mass = wp(303, pimax, nthreads, rp_vals, coms[:,0][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,1][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], coms[:,2][(mass3>=m_bins[i]) & (mass3<m_bins[i+1])], output_rpavg = True)
        ax5[0].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], label = str(m_bins[i])+"-"+str(m_bins[i+1]), c = colours[i])
        
        #spline = UnivariateSpline(wp_300_mass["rmin"], wp_300_mass["wp"], k=3, s = 5)
        #spline = CubicSpline(wp_300_mass["rpavg"], wp_300_mass["wp"])
        spline = UnivariateSpline(wp_300_mass["rpavg"], wp_300_mass["wp"], k=3, s = 100000)
        ax5[0].plot(rp_vals2, spline(rp_vals2), linestyle = "--", zorder=12, color = colours[i])
        #print(wp_300_mass["rpavg"], wp_300_mass["wp"])
        #print(rp_vals, spline(rp_vals))
    
    print("BRIGHT")
    for i in range(1,n_bins):
        wp_300_mass = wp(303, pimax, nthreads, rp_vals, coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,1][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], coms00[:,2][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], output_rpavg = True)
        print(len(coms00[:,0][(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])]))
        #spline = CubicSpline(wp_300_mass["rmin"], wp_300_mass["wp"])
        spline = UnivariateSpline(wp_300_mass["rpavg"], wp_300_mass["wp"], k=3, s = 100000)
        ax5[1].plot(rp_vals2, spline(rp_vals2), linestyle = "--", zorder=12, color = colours[i])
        #wp_err = JK_wp_err(303, pimax, nthreads, bins, coms00[(mass00>=m_bins0[i]) & (mass00<m_bins0[i+1])], volbin_num300)
        ax5[1].errorbar(wp_300_mass["rpavg"], wp_300_mass["wp"], color = colours[i], label = str(m_bins0[i])+"-"+str(m_bins0[i+1]))
        #print(wp_300_mass["rpavg"], wp_300_mass["wp"])
        
ax5[0].set_xscale("log")
ax5[0].set_yscale("log")
ax5[1].set_xscale("log")
ax5[1].set_yscale("log")
ax5[0].set_xlim(0.5,1.5)
ax5[0].legend()
ax5[1].legend()
ax5[1].set_xlim(0.5,1.5)
#ax5[0].set_ylim(1,200)
ax5[1].set_ylim(150,1500)