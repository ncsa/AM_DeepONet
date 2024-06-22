import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from deepxde.backend import tf
tf.config.optimizer.set_jit(True) # This_line_here
import os
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
import keras.backend as K
import time as TT
dde.config.disable_xla_jit()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_loc = './Solidification/DATA'

total_sims = 5500
n_step = 101
n_nodes = 602

HIDDEN = 100
N_input_fn = 2 # Giving two different inputs to the branch network
N_component = 2 # Predict two physical properties, temp and stress
N_output_frame = 1 # predicting the last frame

m = 101
batch_size = 64
seed = 2024
try:
	tf.keras.backend.clear_session()
	tf.keras.utils.set_random_seed(seed)
	tf.random.set_seed(seed)
except:
	pass
dde.config.set_default_float("float64")


class DeepONetCartesianProd(dde.maps.NN):
	"""Deep operator network for dataset in the format of Cartesian product.

	Args:
		layer_sizes_branch: A list of integers as the width of a fully connected network,
			or `(dim, f)` where `dim` is the input dimension and `f` is a network
			function. The width of the last layer in the branch and trunk net should be
			equal.
		layer_sizes_trunk (list): A list of integers as the width of a fully connected
			network.
		activation: If `activation` is a ``string``, then the same activation is used in
			both trunk and branch nets. If `activation` is a ``dict``, then the trunk
			net uses the activation `activation["trunk"]`, and the branch net uses
			`activation["branch"]`.
	"""

	def __init__(
		self,
		layer_sizes_branch,
		layer_sizes_trunk,
		activation,
		kernel_initializer,
		regularization=None,
	):
		super().__init__()
		if isinstance(activation, dict):
			activation_branch = activation["branch"]
			self.activation_trunk = dde.maps.activations.get(activation["trunk"])
		else:
			activation_branch = self.activation_trunk = dde.maps.activations.get(activation)

		# User-defined network
		self.branch = layer_sizes_branch[1]
		self.trunk = layer_sizes_trunk[0]
		# self.b = tf.Variable(tf.zeros(1),dtype=np.float64)
		self.b = tf.Variable(tf.zeros(1, dtype=dde.config.real(tf)))

	def call(self, inputs, training=False):
		x_func = inputs[0]
		x_loc = inputs[1]

		# Branch net to encode the input function
		x_func = self.branch(x_func)

		# Trunk net to encode the domain of the output function
		if self._input_transform is not None:
			x_loc = self._input_transform(x_loc)
		x_loc = self.activation_trunk(self.trunk(x_loc))

		# Dot product
		x = tf.einsum("bht,nhc->btnc", x_func, x_loc)

		# Add bias
		x += self.b

		return tf.math.sigmoid(x) # This_line_different_here_here

class TripleCartesianProd(Data):
	"""Dataset with each data point as a triple. The ordered pair of the first two
	elements are created from a Cartesian product of the first two lists. If we compute
	the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

	This dataset can be used with the network ``DeepONetCartesianProd`` for operator
	learning.

	Args:
		X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
			`dim1`), and the second element has the shape (`N2`, `dim2`).
		y_train: A NumPy array of shape (`N1`, `N2`).
	"""

	def __init__(self, X_train, y_train, X_test, y_test):
		self.train_x, self.train_y = X_train, y_train
		self.test_x, self.test_y = X_test, y_test

		self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
		self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

	def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
		return loss_fn(targets, outputs)

	def train_next_batch(self, batch_size=None):
		if batch_size is None:
			return self.train_x, self.train_y
		if not isinstance(batch_size, (tuple, list)):
			indices = self.branch_sampler.get_next(batch_size)
			return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
		indices_branch = self.branch_sampler.get_next(batch_size[0])
		indices_trunk = self.trunk_sampler.get_next(batch_size[1])
		return (
			self.train_x[0][indices_branch],
			self.train_x[1][indices_trunk],
		), self.train_y[indices_branch, indices_trunk]

	def test(self):
		return self.test_x, self.test_y


def Survey( data ):
	print('Mean ' , np.mean(data) , ' , max ' , np.max(data) , ' , min ' , np.min(data) )


# Exclude extra failed simulations

problem_data = [3222, 4150]
failed_sims = np.load(os.path.join(data_loc, 'failed_sims_time_steps.npy')).astype(np.float64)
failed_sims = np.append(failed_sims, problem_data)

all_sims = np.arange(total_sims)
successful_sims = np.setdiff1d(all_sims, failed_sims).astype(int)
print('failed_sims', failed_sims)

# Number of successful simulations
n_cases = len(successful_sims)


# Load trunk input data and scale
trunk_scaling_factor = 1000
xy_train_testing = np.load(os.path.join(data_loc, 'xy_train_testing.npy')).astype(np.float64)
xy_train_testing = xy_train_testing * trunk_scaling_factor



# Load branch input data
flux = np.loadtxt(os.path.join(data_loc, 'flux_amp.txt'))
disp = np.loadtxt(os.path.join(data_loc, 'disp_amp.txt'))

# Filter all input data based on successful simulations
flux_filtered = flux[successful_sims]
disp_filtered = disp[successful_sims]


# Scale all input data
scalerFlux = MinMaxScaler()
scalerFlux.fit(flux_filtered)
flux_filtered = scalerFlux.transform(flux_filtered)
print( 'flux_filtered' )
Survey( flux_filtered )
print('----------')

scalerDisp = MinMaxScaler()
scalerDisp.fit(disp_filtered)
disp_filtered = scalerDisp.transform(disp_filtered)
print( 'disp_filtered' )
Survey( disp_filtered )
print('----------')

Heat_Amp = np.stack( [flux_filtered,disp_filtered] , axis=-1 )


# Load targets
temp_data = np.load(os.path.join(data_loc, 'last_step_data.npz'))['t']
print('temp_data shape')
print(temp_data.shape)

stress_scaling_factor = 1
stress_data = np.load(os.path.join(data_loc, 'last_step_data.npz'))['s']
stress_data = stress_data / stress_scaling_factor
print('stress_data shape')
print(stress_data.shape)


# Scale targets

data_t = temp_data.copy()
data_s = stress_data.copy()

scalerT = MinMaxScaler()
scalerT.fit(data_t)
scaled_temp = scalerT.transform(data_t)
print('Temp survey: ')
Survey(scaled_temp)
print('----------')

scalerS = MinMaxScaler()
scalerS.fit(data_s)
scaled_stress = scalerS.transform(data_s)
print('Stress survey: ')
Survey(scaled_stress)
print('----------')


Temp = np.zeros((n_cases , N_output_frame , n_nodes , N_component) )
Temp[:, -1, :602, 0] = scaled_temp
Temp[:, -1, :602, 1] = scaled_stress
print('Target shape: ', Temp.shape)


fraction_train = 0.8
print('fraction_train = ' + str(fraction_train) )


# Set count for multiple training for reproducibility
count = 0
print('--------------------------------------')
print('Trial number: ' + str(count))
# Train / test split
N_valid_case = len(Heat_Amp)
N_train = int( N_valid_case * fraction_train )

# Split data for extrapolation
ref_flux = flux_filtered[0]
ref_disp = disp_filtered[0]

flux_L2_dist = np.linalg.norm(flux_filtered - ref_flux, axis=1)
disp_L2_dist = np.linalg.norm(disp_filtered - ref_disp, axis=1)

combined_L2_dist = flux_L2_dist + disp_L2_dist
combined_dist_indices = np.argsort(combined_L2_dist)

test_num = round(flux_filtered.shape[0] * 0.2)
train_num = flux_filtered.shape[0] - test_num

train_case = combined_dist_indices[:train_num]
test_case = combined_dist_indices[train_num:]



u0_train = Heat_Amp[ train_case , :: ]
u0_testing = Heat_Amp[ test_case , :: ]
s_train = Temp[ train_case , : ]
s_testing = Temp[ test_case , : ]


print('u0_train.shape = ',u0_train.shape)
print('type of u0_train = ', type(u0_train))
print('u0_testing.shape = ',u0_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

x_train = (u0_train, xy_train_testing)
y_train = s_train
x_test = (u0_testing, xy_train_testing)
y_test = s_testing
data = TripleCartesianProd(x_train, y_train, x_test, y_test)


my_act1 = "tanh"
branch = tf.keras.models.Sequential([
	 tf.keras.layers.GRU(units=256,batch_input_shape=(batch_size, m, N_input_fn),activation = my_act1,return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
	 tf.keras.layers.GRU(units=128,activation = my_act1,return_sequences = False, dropout=0.00, recurrent_dropout=0.00),
	 tf.keras.layers.RepeatVector(HIDDEN),
	 tf.keras.layers.GRU(units=128,activation = my_act1,return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
	 tf.keras.layers.GRU(units=256,activation=my_act1,return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
	 tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
branch.summary()

my_act2 = "relu"
trunk = tf.keras.models.Sequential([
		tf.keras.layers.InputLayer(input_shape=(2,)),
		tf.keras.layers.Dense(101, activation=my_act2, kernel_initializer='GlorotNormal'),
		tf.keras.layers.Dense(101, activation=my_act2, kernel_initializer='GlorotNormal'),
		tf.keras.layers.Dense(101, activation=my_act2, kernel_initializer='GlorotNormal'),
		tf.keras.layers.Dense(101, activation=my_act2, kernel_initializer='GlorotNormal'),
		tf.keras.layers.Dense(101, activation=my_act2, kernel_initializer='GlorotNormal'),
		tf.keras.layers.Dense( HIDDEN * N_component , activation=my_act2 , kernel_initializer='GlorotNormal'),
		tf.keras.layers.Reshape( [ HIDDEN , N_component ] ),
							  ])
trunk.summary()


# Build model

net = DeepONetCartesianProd(
		[m, branch], [trunk], my_act2, "Glorot normal")

model = dde.Model(data, net)
print("y_train shape:", y_train.shape)


# Define custom loss and error functions
def MSE( y_true, y_pred ):
	#tmp = tf.math.square( K.flatten(Back_to_origin(y_true)) - K.flatten(Back_to_origin(y_pred)) )
	tmp = tf.math.square( K.flatten(y_true) - K.flatten(y_pred) )
	# print('-----Inside MSE function-----')
	# print('y_true type: ', type(y_true))
	# print('y_true: ', y_true)
	# print('y_pred type: ', type(y_pred))
	# print('y_pred: ', y_pred)
	data_loss = tf.math.reduce_mean(tmp)
	return data_loss

def MAE( y_true, y_pred ):
	tmp = tf.math.abs( K.flatten(y_true) - K.flatten(y_pred))
	data_loss = tf.math.reduce_mean(tmp)
	# print('-----Inside MAE function-----')
	# print(data_loss)
	# print('y_true type: ', type(y_true))
	# print('y_true shape: ', tf.shape(y_true))
	# print('y_pred shape: ', tf.shape(y_pred))
	# print('y_true: ', y_true)
	return data_loss

def COP( y_true, y_pred ):
	sqr_err = tf.math.square(K.flatten(y_true) - K.flatten(y_pred))
	var_true = y_true.shape[0] * tf.math.reduce_variance(K.flatten(y_true))
	data_loss = tf.math.divide(tf.math.reduce_sum(sqr_err), var_true)
	return data_loss

def err( y_train , y_pred ):
	ax = 1
	return np.linalg.norm( y_train - y_pred , axis=ax ) / np.linalg.norm( y_train , axis=ax )

def err2( y_train, y_pred ):
	ax = 1
	abs_diff = np.abs( y_train - y_pred)
	test_minus_mean = np.abs(y_train - y_train.mean(axis=ax).reshape(-1, 1))
	rel_abs_err = np.sum(abs_diff, axis=ax) / np.sum(test_minus_mean, axis=ax)
	return rel_abs_err

def err3( y_train, y_pred ):
	ax = 1
	sum_sqr_error = np.sum(np.power(y_train - y_pred, 2), axis=1)
	test_variance = np.sum(np.power(y_train - np.mean(y_train, axis=1).reshape(-1, 1), 2), axis=1)
	Cop_sam = 1 - sum_sqr_error / test_variance
	return Cop_sam

def metric1( y_train , y_pred ):
	y_train_original = scalerS.inverse_transform(y_train[:,0,:,1])
	y_pred_original = scalerS.inverse_transform(y_pred[:,0,:,1])
	return np.mean( err( y_train_original , y_pred_original ).flatten() )

def metric2( y_train, y_pred ):
	y_train_original = scalerS.inverse_transform(y_train[:,0,:,1])
	y_pred_original = scalerS.inverse_transform(y_pred[:,0,:,1])
	return np.mean(err2(y_train_original, y_pred_original).flatten())

def metric3( y_train, y_pred):
	y_train_original = scalerS.inverse_transform(y_train[:,0,:,1])
	y_pred_original = scalerS.inverse_transform(y_pred[:,0,:,1])
	return np.mean(err3(y_train_original, y_pred_original).flatten() )


model.compile(
	"adam",
	lr=1e-3,
	decay=("inverse time", 1, 1e-4),
	loss = COP,
	metrics=[metric1],
)
losshistory, train_state = model.train(iterations=351000, batch_size=batch_size, model_save_path="TrainFrac_"+str(idx) )
np.save('losshistory'+str(idx)+'.npy',losshistory)

st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )


# Transform targets and predicted value to their original scale
Org_temp_test = scalerT.inverse_transform(y_test[:,0,:,0])
Org_stress_test = scalerS.inverse_transform(y_test[:,0,:,1])
Org_temp_pred = scalerT.inverse_transform(y_pred[:,0,:,0])
Org_stress_pred = scalerS.inverse_transform(y_pred[:,0,:,1])


np.savez_compressed('Ver_4_Org_y_try_'+str(count)+'.npz', 
	a=Org_temp_test, 
	b=Org_stress_test, 
	c=Org_temp_pred, 
	d=Org_stress_pred, 
	e=test_case)


# Calculate errors 
error_t = err(y_test[:, 0, :, 0], y_pred[:, 0, :, 0])
error_s = err(y_test[:, 0, :, 1], y_pred[:, 0, :, 1])
org_error_t = err(Org_temp_test, Org_temp_pred)
org_error_s = err(Org_stress_test, Org_stress_pred)

rae_org_error_t = err2(Org_stress_test, Org_stress_pred)
rae_org_error_s = err2(Org_stress_test, Org_stress_pred)


print('$$$$$$$$$$$$$$$$$$$$$Trial num: ' + str(count) + '$$$$$$$$$$$$$$$$$$$$$$$$$$')
print()
print("error_t = ", error_t)
print()
print('----------------------------------------')
print()
print("error_s = ", error_s)
print('----------------------------------------')
print()
print()
print('Stress_scaling_factor: ', stress_scaling_factor)
print('Trunk_scaling_factor: ', trunk_scaling_factor)
print()
print('Scaled L2 error')
print('mean of temperature relative L2 error of s: {:.2e}'.format(error_t.mean()))
print('std of temperature relative L2 error of s: {:.2e}'.format(error_t.std()))
print('--------------------------------------------------------------')
print('mean of stress relative L2 error of s: {:.2e}'.format(error_s.mean()))
print('std of stress relative L2 error of s: {:.2e}'.format(error_s.std()))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------')
print()
print('Origianl L2 error')
print('mean of temperature relative L2 error of s: {:.2e}'.format(org_error_t.mean()))
print('std of temperature relative L2 error of s: {:.2e}'.format(org_error_t.std()))
print('--------------------------------------------------------------')
print('mean of stress relative L2 error of s: {:.2e}'.format(org_error_s.mean()))
print('std of stress relative L2 error of s: {:.2e}'.format(org_error_s.std()))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------')
print()
print('Origianl RAE')
print('mean of temperature relative L2 error of s: {:.2e}'.format(rae_org_error_t.mean()))
print('std of temperature relative L2 error of s: {:.2e}'.format(rae_org_error_t.std()))
print('--------------------------------------------------------------')
print('mean of stress relative L2 error of s: {:.2e}'.format(rae_org_error_s.mean()))
print('std of stress relative L2 error of s: {:.2e}'.format(rae_org_error_s.std()))
print('--------------------------------------------------------------')
print('--------------------------------------------------------------')
print()
print()
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
print()
print()
print()

plt.hist( error_s.flatten() , bins=25 )
plt.savefig('Stress_Err_hist_DeepONet'+str(idx)+'.jpg' , dpi=300)

plt.hist( error_t.flatten() , bins=25 )
plt.savefig('Temp_Err_hist_DeepONet'+str(idx)+'.jpg' , dpi=300)