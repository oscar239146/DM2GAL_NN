from basemodule import *
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import QuantileTransformer

def ReLU(x, maxval=1):
	return keras.backend.relu(x, max_value=maxval)

def offsets(data):
	Offsets = []
	for o in range(np.shape(data)[-1]):
		dat = data[...,o].ravel()
		udat = np.unique(dat)
		diff = np.diff(udat)
		Offsets.append([abs(diff[1] - diff[0]), abs(diff[-1] - diff[-2])])
	return np.asarray(Offsets)


def GQTvecnorm(data):
	'''
	Gaussian Quantile Transformation of 3D (temporal) dataset, with vector normalisation.
	'''
	GQT = QuantileTransformer(output_distribution='normal')
	datashape = np.shape(data)
	sample_ax, timestep_ax, quantity_ax = datashape
	Data = data.reshape(sample_ax*timestep_ax, quantity_ax)
	qData = GQT.fit_transform(Data)
	Offsets = offsets(qData)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] += Offsets[o,0]
		qData[wmax] -= Offsets[o,1]
	qData = qData.reshape(*datashape)
	return qData, GQT, Offsets

def GQTscalnorm(data, add_dim=False):
	'''
	Gaussian Quantile Transformation of 2D (non-temporal) or 3D (temporal) dataset, with scalar normalisation.
	'''
	GQT = QuantileTransformer(output_distribution='normal')
	datashape = np.shape(data)
	if len(datashape) > 2:
		sample_ax, timestep_ax, quantity_ax = datashape
		Data = data.reshape(sample_ax, quantity_ax*timestep_ax)
		qData = GQT.fit_transform(Data)
		Offsets = offsets(qData)
		for o in range(len(Offsets)):
			wmin = np.where(qData[...,o]==np.min(qData[...,o]))
			wmax = np.where(qData[...,o]==np.max(qData[...,o]))
			qData[wmin] += Offsets[o,0]
			qData[wmax] -= Offsets[o,1]
		qData = qData.reshape(*datashape)
	else:
		qData = GQT.fit_transform(data)
		Offsets = offsets(qData)
		for o in range(len(Offsets)):
			wmin = np.where(qData[...,o]==np.min(qData[...,o]))
			wmax = np.where(qData[...,o]==np.max(qData[...,o]))
			qData[wmin] += Offsets[o,0]
			qData[wmax] -= Offsets[o,1]
		if add_dim:
			qData = qData.reshape(*datashape, 1)
	return qData, GQT, Offsets

def GQTvecinv(qData, GQT, Offsets):
	'''
	Inverse Gaussian Quantile Transformation of 3D (temporal) dataset and GQT object, with vector normalisation.
	'''
	datashape = np.shape(qData)
	sample_ax, timestep_ax, quantity_ax = datashape
	qData = qData.reshape(sample_ax*timestep_ax, quantity_ax)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] -= Offsets[o,0]
		qData[wmax] += Offsets[o,1]
	data = GQT.inverse_transform(qData)
	return data.reshape(*datashape)

def GQTscalinv(qData, GQT, Offsets, add_dim=False):
	'''
	Inverse Gaussian Quantile Transformation of 2D (non-temporal) or 3D (temporal) dataset and GQT object, with scalar normalisation.
	'''
	datashape = np.shape(qData)
	for o in range(len(Offsets)):
		wmin = np.where(qData[...,o]==np.min(qData[...,o]))
		wmax = np.where(qData[...,o]==np.max(qData[...,o]))
		qData[wmin] -= Offsets[o,0]
		qData[wmax] += Offsets[o,1]
	if len(datashape) > 2:
		sample_ax, timestep_ax, quantity_ax = datashape
		qData = qData.reshape(sample_ax, quantity_ax*timestep_ax)
		data = GQT.inverse_transform(qData)
		return data.reshape(*datashape)
	else:
		data = GQT.inverse_transform(qData)
		if add_dim:
			qData = qData.reshape(*datashape, 1)
		return data

def lr_exp(E, lr0=0.001, k=0.1):
	'''
	Variable exponential learning rate, for recursive use in network training.
	'''
	if E > 0:
		lr0 *= np.exp(-k)
	if lr0<=1e-8:
		lr0 = 1e-6
	return lr0

def lr_expcap(E, lr0=0.0005, k=0.01, cap=1e-7):
	'''
	Capped exponential learning rate. Similar to lr exp, but with a minimum allowed value of the learning rate.
	'''
	if E > 0:
		lr0 *= np.exp(-k)
	return max([lr0, cap])

def lr_step(E, lr0=0.001, dr=0.5, Ed=10):
	'''
	Step-based variable learning rate. Learning rate is multiplied by a constant r at equally spaced intervals, E_d epochs apart.
	'''
	if E % Ed == 0 and E > 0:
		lr0 *= dr
	return lr0

def lr_step_r(E, lr0=0.001, dr=0.5, rp=0.25, Es=10):
	'''
	Random step function. For each epoch after a specified number, there is a probability that the learning rate is randomly multiplied by the step factor.
	'''
	rn = np.random.uniform()
	if rn <= rp and E > Es:
		lr0 *= dr
	return lr0

def dark_central_inv():
	def sort_arrays(arrays):
		arrs = zip(*sorted(zip(*arrays)))
		Arrs = np.asarray([np.asarray(arr) for arr in arrs])
		return Arrs

	def func(x, y):
		X = x.ravel()
		Y = y.ravel()
		X, Y = sort_arrays((X, Y))
		X, W = np.unique(X, return_index=True)
		Y = Y[W]
		fwd_interp = interp1d(X, Y, bounds_error=False, fill_value=(Y[0], Y[-1]), kind='linear')
		inv_interp = interp1d(Y, X, bounds_error=False, fill_value=(X[0], X[-1]), kind='linear')
		return fwd_interp, inv_interp

	def transform(x, y, data):
		fwd_func, inv_func = func(x, y)
		transformed = fwd_func(data)
		return transformed, inv_func

	Datac = h5py.File('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/central_nndata_dark.h5', 'r')

	datac1 = np.asarray([Datac['Mhdot'][:].T, Datac['delta1'][:].T, Datac['delta3'][:].T, Datac['delta5'][:].T, Datac['vcirc'][:].T, Datac['rhalf'][:].T, Datac['skew'][:].T, Datac['minD'][:].T]).T
	datac2 = np.asarray([Datac['beta'][:], Datac['d_min'][:], Datac['d_node'][:], Datac['d_saddle_1'][:], Datac['d_saddle_2'][:], Datac['d_skel'][:], Datac['formtime'][:], Datac['logMh'][:], Datac['logmaxMhdot'][:]]).T
	datac = np.load('/home/oscar239146/Uni/NN_Thesis/Chittenden_Data/centralmodeldata.npz')

	scalc = [1, 2, 3]
	Datac1 = []
	inverse_funcs1 = []

	x1c = datac['x1']
	x1qc = datac['x1q']
	x2c = datac['x2']
	x2qc = datac['x2q']
	for i in range(np.shape(x1c)[2]):
		transdata = datac1[:,:,i]
		x1 = x1c[:,:,i]
		x1q = x1qc[:,:,i]
		if i not in scalc:
			transformed, inv_f = transform(x1, x1q, transdata)
			#Datac1.append(transform(x1, x1q, transdata))
			Datac1.append(transformed)
			inverse_funcs1.append(inv_f)
		else:
			Datac1ij = []
			inv_funcsij = []
			for j in range(np.shape(x1c)[1]):
				transformed, inv_f = transform(x1[:, j], x1q[:, j], transdata[:, j])
				#Datac1ij.append(transform(x1[:,j], x1q[:,j], transdata[:,j]))
				Datac1ij.append(transformed)
				inv_funcsij.append(inv_f)
			Datac1.append(np.asarray(Datac1ij).T)
			inverse_funcs1.append(inv_funcsij)
	Datac1 = np.asarray(Datac1)
	Datac1 = np.moveaxis(Datac1, 0, -1)
	inverse_funcs1 = np.asarray(inverse_funcs1)
	#Datac2 = np.asarray([transform(x2c[:,i], x2qc[:,i], datac2[:,i]) for i in range(np.shape(x2c)[1])]).T
	Datac2, inverse_funcs2 = zip(*[transform(x2c[:, i], x2qc[:, i], datac2[:, i]) for i in range(np.shape(x2c)[1])])
	Datac2 = np.asarray(Datac2).T
	return inverse_funcs1, inverse_funcs2

seed = 23
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

EPOCHS = int(1 + np.log(1000)/0.1)
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,restore_best_weights=True)

class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', patience=15, restore_best_weights=True, start_epoch=6, verbose=0):
        # Pass arguments to EarlyStopping properly
        super(CustomStopper, self).__init__(
            monitor=monitor, 
            patience=patience, 
            restore_best_weights=restore_best_weights,
            verbose = verbose
        )
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            super().on_epoch_end(epoch, logs)

early_stop = CustomStopper(monitor='val_loss', patience=10, restore_best_weights=True, start_epoch=6, verbose=1)
early_stop_retrain = CustomStopper(monitor='val_loss', patience=5, restore_best_weights=True, start_epoch=6, verbose=1)
#scheduler = keras.callbacks.LearningRateScheduler(lr_expcap, verbose=0)
scheduler = keras.callbacks.LearningRateScheduler(lr_exp, verbose=0)

scheduler_retrain = keras.callbacks.LearningRateScheduler(lr_expcap, verbose=0)

early_stop_sat = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lr_adjust= tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2,verbose=1,mode="auto", min_lr = 1e-7, cooldown =1)
lr_adjust_retrain= tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2,verbose=1,mode="auto", min_lr = 1e-8)

drop_rate = 0.2#0.25
l2_val = 1e-4

def SatelliteNetwork_model12_300():
	'''
	model12 network
	'''
	input1 = keras.Input(shape=(33,7))
	input2 = keras.Input(shape=(11,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros())(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(input2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(conc)
	dense = keras.layers.Dropout(rate=0.25)(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	#dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	#dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5*0.1))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7*0.1))(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5*0.1))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7*0.1))(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5*0.1))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7*0.1))(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val*0.1))(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, kernel_initializer='he_normal')(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.001)
	#optimizer = keras.optimizers.RMSprop(0.0007)
	
	#model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	model.compile(loss=tf.keras.losses.Huber(delta=0.6), optimizer=optimizer, metrics=['mse', 'mae', 'accuracy'])
	
	return model



def CentralNetwork_model13_300():
	'''
	Compiles the network designed for use with central galaxies.
	Best model TNG300
	'''
	input1 = keras.Input(shape=(33,8))
	input2 = keras.Input(shape=(9,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(),return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	#rnn = keras.layers.Dropout(rate=0.1)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	#rnn = keras.layers.Dropout(rate=0.1)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=False)(rnn)
	#rnn = keras.layers.Dropout(rate=0.1)(rnn)
 
	#t_dist = keras.layers.TimeDistributed(keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal'))(rnn)
	#t_dist = keras.layers.GlobalAveragePooling1D()(t_dist)
 
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	#dense1 = keras.layers.Dropout(rate=0.4)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	#dense1 = keras.layers.Dropout(rate=0.4)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	#dense1 = keras.layers.Dropout(rate=0.4)(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(input2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#dense2 = keras.layers.Dropout(rate=0.4)(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(conc)
	dense = keras.layers.Dropout(rate=0.25)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	#dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	#dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7))(dense)
	
 
	#dense = keras.layers.Dropout(rate=0.2)(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7))(dense)
 
	#dense = keras.layers.Dropout(rate=0.2)(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(1e-5))(dense)
	dense = keras.layers.PReLU(alpha_regularizer=keras.regularizers.l2(1e-7))(dense)
 
	#dense = keras.layers.Dropout(rate=0.2)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)# kernel_regularizer=keras.regularizers.l2(0.00005))(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	#dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	#dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.GaussianDropout(rate=drop_rate)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l2_val))(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#dense = keras.layers.Dropout(rate=0.4)(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, kernel_initializer='he_normal')(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.0007)
	#optimizer = keras.optimizers.Adam(0.0007)
	#optimizer = keras.optimizers.RMSprop(0.00035)
	
	
	#model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	model.compile(loss=tf.keras.losses.Huber(delta=0.6), optimizer=optimizer, metrics=['mse', 'mae', 'accuracy'], weighted_metrics =['mse'])
	
	return model



def SatelliteNetwork_model5_100():
	'''
	Compiles the network designed for use with TNG100 satellite galaxies.
	Used for model5.
	'''
	input1 = keras.Input(shape=(33,7))
	input2 = keras.Input(shape=(11,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(),)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(input2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(conc)
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, kernel_initializer='he_normal')(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.001)
	#optimizer = keras.optimizers.RMSprop(0.0007)
	
	#model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	model.compile(loss=tf.keras.losses.Huber(delta=0.6), optimizer=optimizer, metrics=['mae'])
	
	return model

def CentralNetwork_model5_100():
	'''
	Compiles the network designed for use with central galaxies.
	Used for model5.
	'''
	input1 = keras.Input(shape=(33,8))
	input2 = keras.Input(shape=(9,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(),return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros(), return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, kernel_initializer=keras.initializers.glorot_uniform(), recurrent_initializer=keras.initializers.orthogonal(), bias_initializer=keras.initializers.zeros())(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu, kernel_initializer='he_normal')(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(input2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu, kernel_initializer='he_normal')(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(conc)
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(48, kernel_initializer='he_normal')(dense)
	dense = keras.layers.PReLU()(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)#, kernel_regularizer=keras.regularizers.l2(0.0001))(dense)
	
	dense = keras.layers.Dense(51, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)# kernel_regularizer=keras.regularizers.l2(0.00005))(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear, kernel_initializer='he_normal')(dense)
	dense = keras.layers.Dense(69, kernel_initializer='he_normal')(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	#optimizer = keras.optimizers.RMSprop(0.0007)
	optimizer = keras.optimizers.RMSprop(0.00035)
	
	#model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	model.compile(loss=tf.keras.losses.Huber(delta=0.725), optimizer=optimizer, metrics=['mae'])
	
	return model

def SatelliteNetworko():
	'''
	Compiles the network designed for use with satellite galaxies.
	'''
	input1 = keras.Input(shape=(33,7))
	input2 = keras.Input(shape=(11,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(input2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(11, activation=tf.nn.elu)(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(conc)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(44, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear)(dense)
	dense = keras.layers.Dense(69)(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.001)
	
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	
	return model


def CentralNetworko():
	'''
	Compiles the network designed for use with central galaxies.
	'''
	input1 = keras.Input(shape=(33,8))
	input2 = keras.Input(shape=(9,))
	
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(input1)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu, return_sequences=True)(rnn)
	rnn = keras.layers.SimpleRNN(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(rnn)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	dense1 = keras.layers.Dense(33, activation=tf.nn.elu)(dense1)
	model1 = keras.Model(inputs=input1, outputs=dense1)
	
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(input2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	#drop2 = keras.layers.Dropout(rate=0.05)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	dense2 = keras.layers.Dense(9, activation=tf.nn.elu)(dense2)
	model2 = keras.Model(inputs=input2, outputs=dense2)
	
	inp = [model1.input, model2.input]
	conc = keras.layers.concatenate([model1.output, model2.output])
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(conc)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(42, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(45, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(48, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(51, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(54, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(57, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	#drop = keras.layers.Dropout(rate=0.05)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(60, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(63, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(66, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=tf.nn.elu)(dense)
	dense = keras.layers.Dense(69, activation=keras.activations.linear)(dense)
	dense = keras.layers.Dense(69)(dense)
	model = keras.Model(inputs=inp, outputs=dense)
	
	optimizer = keras.optimizers.RMSprop(0.0007)
	
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	
	return model