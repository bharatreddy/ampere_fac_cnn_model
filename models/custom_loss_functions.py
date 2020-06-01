from keras import backend as K
from keras.layers import Multiply, Reshape
import numpy as np
import tensorflow as tf
from functools import partial

def rmse_w(y_true, y_pred, weights):
    #return K.sqrt(K.mean(K.square(y_true - y_pred)))
    return K.sqrt(K.mean(tf.multiply(weights, K.square(y_true - y_pred))))
    #return K.sqrt(K.mean(Multiply()([weights, K.square(y_true - y_pred)])))

def mse_w(y_true, y_pred, weights):
    #return K.mean(K.square(y_true - y_pred)) 
    #import pdb
    #pdb.set_trace() 

    return K.mean(tf.multiply(weights, K.square(y_true - y_pred))) 
    #return K.mean(Multiply()([weights, K.square(y_true - y_pred)])) 

def mae_w(y_true, y_pred, weights):
    #return K.mean(K.abs(y_true - y_pred)) 
    #return K.mean(Multiply()([weights, K.abs(y_true - y_pred)])) 
    return K.mean(tf.multiply(weights, K.abs(y_true - y_pred))) 

def cce(y_true, y_pred):
    x = tf.convert_to_tensor(y_true, np.float32)
    y = tf.convert_to_tensor(y_pred, np.float32)
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

# Consturct masking matrix
#W = np.ones((60, 24), dtype="float32")
#W[:20, :] = 0
W = np.ones((50, 24), dtype="float32")
W[:10, :] = 0
W = W.reshape((1, -1))

# Another set of weights could be from the average matrix derived from AMPERE data
avg_weights = np.load("../params/jr_magn_median_2011_2015_.npy")
# Normalize the weights so they are between 0,1
avg_weights = avg_weights/avg_weights.max()
avg_weights = avg_weights.reshape((1, -1))
avg_weights = np.float32(avg_weights)


# Contruct partial functions to pass the masking matrix
rmse = partial(rmse_w, weights=W)
rmse.__name__ = "rmse"
mse = partial(mse_w, weights=W)
mse.__name__ = "mse"
mae = partial(mae_w, weights=W)
mae.__name__ = "mae"
# another partial with average weights
rmse_med_jr = partial(rmse_w, weights=avg_weights)
rmse_med_jr.__name__ = "rmse_med_jr"
mae_med_jr = partial(mae_w, weights=avg_weights)
mae_med_jr.__name__ = "mae_med_jr"

if __name__ == "__main__":

    y_true = np.load("../data/ampere/north_20110429.2200.npy")
    y_pred = y_true + 0.001 * np.random.randn(y_true.shape[0], y_true.shape[1]) 
    y_true = y_true.reshape((1, -1))
    y_pred = y_pred.reshape((1, -1))

    rms = rmse(y_true, y_pred)
    ms = mse(y_true, y_pred)
    ma = mae(y_true, y_pred)
    cc = cce(y_true, y_pred)
    print(rms)
    print(ms)
    print(ma)
    print(cc)

