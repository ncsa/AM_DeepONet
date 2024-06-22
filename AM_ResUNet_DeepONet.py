import sys
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
from deepxde.backend import tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import math
import tensorflow as tf
import time as TT

tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(True)

class ResBlock(Layer):
    """
    Represents the Residual Block in the ResUNet architecture.
    """
    def __init__(self, filters, strides, l2 , **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.l2 = l2

        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", use_bias=False,kernel_regularizer=l2)

        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False,kernel_regularizer=l2)

        self.conv_skip = Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False,kernel_regularizer=l2)
        self.bn_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        skip = self.conv_skip(inputs)
        skip = self.bn_skip(skip, training=training)

        res = self.add([x, skip])
        return res

    def get_config(self):
        return dict(filters=self.filters, strides=self.strides, **super(ResBlock, self).get_config())

def ResUNet(input_shape , classes , filters_root, depth, drop_rate , L2_reg ):
    """
    Builds ResUNet model.
    :param input_shape: Shape of the input images (h, w, c). Note that h and w must be powers of 2.
    :param classes: Number of classes that will be predicted for each pixel. Number of classes must be higher than 1.
    :param filters_root: Number of filters in the root block.
    :param depth: Depth of the architecture. Depth must be <= min(log_2(h), log_2(w)).
    :return: Tensorflow model instance.
    """
    regularizer = tf.keras.regularizers.L2(L2_reg)

    if not math.log(input_shape[0], 2).is_integer() or not math.log(input_shape[1], 2):
        raise ValueError(f"Input height ({input_shape[0]}) and width ({input_shape[1]}) must be power of two.")
    if 2 ** depth > min(input_shape[0], input_shape[1]):
        raise ValueError(f"Model has insufficient height ({input_shape[0]}) and width ({input_shape[1]}) compared to its desired depth ({depth}).")

    # Load parameters
    input2 = Input(shape=2) # load paras
    layer2 = Dense( 25 )( input2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( 25 )( layer2 )
    encoded_load = Dense( 64*64 )( layer2 )


    input1 = Input(shape=input_shape[0]*input_shape[1]) # Geom
    layer = tf.keras.layers.Reshape( (input_shape[0],input_shape[1],1 ) )(input1)


    # ENCODER
    encoder_blocks = []

    filters = filters_root
    layer = Conv2D(filters=filters, kernel_size=7, strides=1, padding="same",kernel_regularizer=regularizer)(layer)

    branch = Conv2D(filters=filters, kernel_size=7, strides=1, padding="same", use_bias=False,kernel_regularizer=regularizer)(layer)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)
    branch = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", use_bias=True,kernel_regularizer=regularizer)(branch)

    layer = Add()([branch, layer])

    encoder_blocks.append(layer)

    for _ in range(depth - 1):
        filters *= 2
        layer = ResBlock(filters, strides=2,l2=regularizer)(layer)
        layer = Dropout( drop_rate )(layer)
        encoder_blocks.append(layer)

    # BRIDGE
    filters *= 2
    layer = ResBlock(filters, strides=2,l2=regularizer)(layer)

    ##################################################################
    # This is the smallest point
    encoded_geom = tf.keras.layers.Flatten()( layer )


    # ##################################################################
    # # Data fusion
    # print('________Data fusion__________')
    # print(encoded_geom.shape)
    # print(encoded_load.shape)

    mixed = tf.math.multiply( encoded_geom , encoded_load )
    layer = tf.keras.layers.Reshape( (4,4,256) )( mixed ) 

    # geometry DECODER
    for i in range(1, depth + 1):
        filters //= 2
        skip_block_connection = encoder_blocks[-i]

        layer = UpSampling2D( interpolation="bilinear" )(layer)
        # layer = Concatenate()([layer, skip_block_connection])
        layer_a = ResBlock(filters, strides=1,l2=regularizer)(layer)
        layer_b = ResBlock(filters, strides=1,l2=regularizer)(skip_block_connection)
        layer = Add()([layer_a, layer_b])

        layer = Dropout( drop_rate )(layer)

    layer = Conv2D(filters=classes, kernel_size=1, strides=1, padding="same",kernel_regularizer=regularizer)(layer)
    layer = Activation(activation="sigmoid")(layer)
    output1 = tf.keras.layers.Reshape( (input_shape[0]*input_shape[1],classes ) )(layer)


    # Load decoder
    layer2 = Dense( 50 )( mixed )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 100 )( layer2 )
    layer2 = Dense( 50 )( layer2 )
    layer2 = Dense( classes*2 )( layer2 )
    output2 = Activation(activation="relu")(layer2)
    output2 = tf.keras.layers.Reshape( (2, classes) )(output2)


    # output = layer
    return Model( [input1,input2] , [output1,output2] )

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

        # self.branch = layer_sizes_branch[1]
        self.trunk = layer_sizes_trunk[1] 
        self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0] # [ bs , 2 ] , load parameters
        x_loc = inputs[1] # [ bs , 128*128 ] , input geometry

        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)

        x_loc, x_func = self.trunk( [x_loc,x_func] )

        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            print('x_func.shape: ', x_func.shape)
            print('x_loc.shape: ', x_loc.shape)
            
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = tf.einsum("imk,ijk->ijm", x_func, x_loc)

        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return tf.math.sigmoid(x)


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
            return (self.train_x[0][indices], self.train_x[1][indices]), self.train_y[indices]
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

def dice_coefficient(A, B):
    """
    Calculate the Dice similarity coefficient between two flattened numpy arrays.

    Parameters:
    A (numpy array): Flattened array A.
    B (numpy array): Flattened array B.

    Returns:
    float: Dice similarity coefficient.
    """
    intersection = np.sum(A & B)
    print(intersection)
    return 2.0 * intersection / (np.sum(A) + np.sum(B))

########################################################################################################
seed = 2024 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

# Parameters
m = [ 64 , 64 ] # Number of geometry parameters
N_load_paras = 2 # Number of load parameters
HIDDEN = 32
DEPTH = 4
FILTER = 16
fraction_train = 0.8
data_type = np.float32
drop_rate = 0.02
L2_reg = 5e-3
sub = '_mixed_DeepONet_LR_5e-4'

print('\n\nModel parameters:')
print( sub )
print( 'HIDDEN  ' , HIDDEN )
print( 'DEPTH  ' , DEPTH )
print( 'FILTER  ' , FILTER )
print( 'fraction_train  ' , fraction_train )
print('\n\n\n')

# Trunk network, for encoding the geometry
trunk = ResUNet(input_shape=(m[0], m[1], 1), classes=HIDDEN, filters_root=FILTER, depth=DEPTH , drop_rate=drop_rate , L2_reg=L2_reg )
print('\n\nTrunk network:')
trunk.summary()


# Build DeepONet
activation = "relu"
net = DeepONetCartesianProd(
        [ N_load_paras , None ], [ m[0]*m[1] , trunk ] , activation, "Glorot normal")


# Load raw input data

nnn = 1000
variation = 5

base = './AM/DATA/'

tmp = np.load(base+'reduced_design_bin.npz')['data']
data_raw = []

File_filter = np.load(base+'file_mask.npy')
for i in range(nnn):
    for j in range(variation):
        if File_filter[i][j]:
            data_raw.append(tmp[i])
            
data_raw = np.array(data_raw)


LoadInfo1 = np.load(base+'filtered_InputsCurves.npy')[:, 0].reshape(-1, 1)
LoadInfo2 = np.load(base+'filtered_GeoVel.npy')


# Scale Inputs 

scalerPower = MinMaxScaler()
scalerPower.fit(LoadInfo1)
LoadInfo1 = scalerPower.transform(LoadInfo1)
print( 'power_filtered' )
Survey( LoadInfo1 )
print('----------')

scalerVel = MinMaxScaler()
scalerVel.fit(LoadInfo2)
LoadInfo2 = scalerVel.transform(LoadInfo2)
print( 'velocity_filtered' )
Survey( LoadInfo2 )
print('----------')

LoadInfo = np.concatenate((LoadInfo1, LoadInfo2), axis=1) # shape becomes (9216, 2)


# Load targets

stress = np.load(base+'smooth_concat_ele_last_step_density_mul_binary.npz')['s']
temp = np.load(base+'smooth_concat_ele_last_step_density_mul_binary.npz')['t']

print( 'Data_raw shape: ', data_raw.shape )
print( 'LoadInfo shape: ', LoadInfo.shape )
print( 'Stress shape: ', stress.shape )
print( 'Temp shape: ', temp.shape )
print()

# Scale targets
scalerS = MinMaxScaler()
scalerS.fit(stress)
scaled_stress = scalerS.transform(stress)
print('Stress survey: ')
Survey(scaled_stress)
print('----------')

scalerT = MinMaxScaler()
scalerT.fit(temp)
scaled_temp = scalerT.transform(temp)
print('Temp survey: ')
Survey(scaled_temp)
print('----------')

stress = scaled_stress
temp = scaled_temp


run_count = 0
print('--------------------------------------------------------------------------------')
# Train / test split
N_valid_case = len(data_raw)
N_train = int( N_valid_case * fraction_train )

design_used = tmp[:1000, :]
standard_des = design_used[0]

tot_dice_coefficient = []
for i in range(len(data_raw)):
    tot_dice_coefficient.append(dice_coefficient(standard_des, data_raw[i]))


tot_dice_coefficient = np.array(tot_dice_coefficient)
sorting_array = 1 - tot_dice_coefficient
des_dist_indices = np.argsort(sorting_array)

train_case = des_dist_indices[:N_train]
test_case = des_dist_indices[N_train:]

# Branch: load parameters
u0_train = LoadInfo[ train_case , :: ].astype(data_type)
u0_testing = LoadInfo[ test_case , :: ].astype(data_type)

# Trunk: geometry
xy_train = data_raw[ train_case , :: ].astype(data_type)
xy_testing = data_raw[ test_case , :: ].astype(data_type)

# Output: stress and temperature
s_train = stress[ train_case , : ].astype(data_type)
s_testing = stress[ test_case , : ].astype(data_type)
t_train = temp[ train_case , : ].astype(data_type)
t_testing = temp[ test_case , : ].astype(data_type)

tot_train = np.stack((s_train, t_train), axis=2)         # Shape becomes (train_case, 4096, 2)
tot_testing = np.stack((s_testing, t_testing), axis=2)   # Shape becomes (testing_case, 4096, 2)

print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('xy_train.shape = ',xy_train.shape)
print('xy_testing.shape = ',xy_testing.shape)
print('tot_train.shape = ',tot_train.shape)
print('tot_testing.shape = ',tot_testing.shape)


# ###################################################################################
# s0_plot = s_train.flatten()
# s1_plot = s_testing.flatten()
# plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
# plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
# plt.legend(['Training' , 'Testing'])
# plt.savefig('train_test_stress_dist.pdf')
# plt.close()
# ###################################################################################
# t0_plot = t_train.flatten()
# t1_plot = t_testing.flatten()
# plt.hist( t0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
# plt.hist( t1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
# plt.legend(['Training' , 'Testing'])
# plt.savefig('train_test_temperature_dist.pdf')
# plt.close()
# ###################################################################################


# Pack data
x_train = (u0_train.astype(data_type), xy_train.astype(data_type))
y_train = tot_train.astype(data_type) 
x_test = (u0_testing.astype(data_type), xy_testing.astype(data_type))
y_test = tot_testing.astype(data_type)
data = TripleCartesianProd(x_train, y_train, x_test, y_test)

# Build model
model = dde.Model(data, net)

# Define metrics and functions

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


def st_metric( y_train , y_pred ):
    true_vals = y_train[:, :, 0]
    pred_vals = y_pred[:, :, 0]

    flag = ( true_vals < 1e-8 )
    pred_vals[ flag ] = 0

    err = []
    for i in range(len(true_vals)):
        error_s_tmp = np.linalg.norm(true_vals[i] - pred_vals[i] ) / np.linalg.norm( true_vals[i] )
        err.append( error_s_tmp )
    return np.mean( err )

def tm_metric( y_train , y_pred ):
    true_vals = y_train[:, :, 1]
    pred_vals = y_pred[:, :, 1]

    flag = ( true_vals < 1e-8 )
    pred_vals[ flag ] = 0

    err = []
    for i in range(len(true_vals)):
        error_s_tmp = np.linalg.norm(true_vals[i] - pred_vals[i] ) / np.linalg.norm( true_vals[i] )
        err.append( error_s_tmp )
    return np.mean( err )

def relativeDiff( y_true, y_pred ):
    diff = y_true - y_pred
    y_true_f = K.flatten(y_true)
    diff_f = K.flatten(diff)
    return tf.norm( diff ) / tf.norm( y_true_f )


model.compile(
    "adam",
    lr=5e-4,
    decay=("inverse time", 1, 1e-4),
    # loss=relativeDiff,
    loss = MSE,
    metrics=[ st_metric, tm_metric ],
)
losshistory, train_state = model.train(iterations=150000, batch_size=128)

st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )


# Convert targets and predicted values to their original scale
org_s_test = scalerS.inverse_transform(y_test[:, :, 0])
org_t_test = scalerT.inverse_transform(y_test[:, :, 1])

org_s_pred = scalerS.inverse_transform(y_pred[:, :, 0])
org_t_pred = scalerT.inverse_transform(y_pred[:, :, 1])

org_y_test = np.zeros(y_test.shape)
org_y_test[:, :, 0] = org_s_test
org_y_test[:, :, 1] = org_t_test

org_y_pred = np.zeros(y_pred.shape)
org_y_pred[:, :, 0] = org_s_pred
org_y_pred[:, :, 1] = org_t_pred


# Calculate error from predicted values
error_s = []
error_t = []
rho = data.test_x[0]
for i in range(len(y_test)):
    s_true_vals = org_s_test[i]
    s_pred_vals = org_s_pred[i]
    t_true_vals = org_t_test[i]
    t_pred_vals = org_t_pred[i]

    flag = ( s_true_vals < 1e-8 )
    s_pred_vals[ flag ] = 0

    error_s_tmp = np.linalg.norm(s_true_vals - s_pred_vals ) / np.linalg.norm( s_true_vals )
    error_t_tmp = np.linalg.norm(t_true_vals - t_pred_vals ) / np.linalg.norm( t_true_vals )

    if error_s_tmp > 1:
        error_s_tmp = 1

    if error_t_tmp > 1:
        error_t_tmp = 1

    error_s.append(error_s_tmp)
    error_t.append(error_t_tmp)
error_s = np.array(error_s)
error_t = np.array(error_t)
print("error_s = ", error_s)
print("error_t = ", error_t)

np.savez_compressed('New_TestData'+sub+'_'+str(run_count)+'.npz',a=org_y_test,b=org_y_pred,c=u0_testing,d=xy_testing,e=train_case,f=test_case,g=losshistory,h=train_state,i=error_s, j=error_t)


print('mean of relative L2 error of s: {:.2e}'.format( np.mean(error_s) ))
print('std of relative L2 error of s: {:.2e}'.format( np.std(error_s) ))
print()
print('mean of relative L2 error of t: {:.2e}'.format( np.mean(error_t) ))
print('std of relative L2 error of t: {:.2e}'.format( np.std(error_t) ))



plt.hist( error_s.flatten() , bins=25 )
plt.savefig('S_Err_hist_DeepONet'+sub+'_'+str(run_count)+'.jpg' , dpi=1000)
plt.hist( error_t.flatten() , bins=25 )
plt.savefig('T_Err_hist_DeepONet'+sub+'_'+str(run_count)+'.jpg' , dpi=1000)
