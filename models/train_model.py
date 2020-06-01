import keras
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from cnn_models import CNN_MLP, train_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
import datetime as dt
import os
import glob
import time
import sys
sys.path.append("../data_processing/")
from create_datapoints import create_datapoints
from model_utils import load_param_data, create_results_folder
from custom_loss_functions import mse, rmse, mae, cce#mean_squared_error, mean_absolute_error
#from custom_loss_functions import root_mean_squared_error
#from custom_loss_functions import correlation_coefficient_loss

# Load the parameters
params_dir = "../params/"
general_params_file = "general.json"
model_params_file = "cnn_model.json"
results_dir="../data/trained_models/"
results_mapper_file="../data/trained_models/mapper.txt"

# load the parameters dict
params_dict = load_param_data(params_dir=params_dir,
  			      model_params_file=model_params_file,
			      general_params_file=general_params_file)

# create a results folder
out_dir = create_results_folder(params_dict, results_dir=results_dir,
                                results_mapper_file=results_mapper_file)

# Initialize parameters
# Parameters needed for constructing data points
start_datetime = dt.datetime.strptime(params_dict["general_params"]["start_datetime"], "%Y-%m-%d %H:%M:%S")
end_datetime = dt.datetime.strptime(params_dict["general_params"]["end_datetime"], "%Y-%m-%d %H:%M:%S")

dataset_dir = params_dict["general_params"]["dataset_dir"]
omn_dbdir = params_dict["general_params"]["omn_dbdir"]
omn_db_name = params_dict["general_params"]["omn_db_name"]
omn_table_name = params_dict["general_params"]["omn_table_name"]

omn_history = params_dict["general_params"]["omn_history"]
missing_omn_maxlim = params_dict["general_params"]["missing_omn_maxlim"]
omn_train_params = params_dict["general_params"]["omn_train_params"]
omn_time_delay = params_dict["general_params"]["omn_time_delay"]
imf_normalize = params_dict["general_params"]["imf_normalize"]
omn_time_resolution = params_dict["general_params"]["omn_time_resolution"]

hemi = params_dict["general_params"]["hemi"]
ampere_time_res = params_dict["general_params"]["ampere_time_res"]
ampere_dir = params_dict["general_params"]["ampere_dir"]

# Parameters needed for creating a model
input_params = params_dict["general_params"]["input_params"]
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "au":5, "al":6}
input_cols = [param_col_dict[x] for x in input_params]

optimizer = params_dict["model_params"]["optimizer"]
learning_rate = params_dict["model_params"]["learning_rate"]
loss = params_dict["model_params"]["loss"]
hidden_layer_activation = params_dict["model_params"]["hidden_layer_activation"]

n_epochs = params_dict["model_params"]["n_epochs"]
batch_size = params_dict["model_params"]["batch_size"]
metrics = params_dict["model_params"]["metrics"]
train_size = params_dict["general_params"]["train_size"]
val_size = params_dict["general_params"]["val_size"]
test_size = params_dict["general_params"]["test_size"]

skip_training = False
#skip_training = True

save_prediction = params_dict["model_params"]["save_prediction"]

# Construct data points
print("loading data points...")
dataset_fname = hemi + "_" + start_datetime.strftime("%Y%m%d") +\
                "_" + end_datetime.strftime("%Y%m%d") + ".npz"
dataset_file = os.path.join(dataset_dir, dataset_fname)
datapoints = np.load(dataset_file)

# Select certain columns
X = datapoints["X"]
X = X[:, :, input_cols]
Y = datapoints["Y"]
xy_time = datapoints["time"]

###################
# For testing
#X = Y[:-1, :]
#Y = Y[1:, :]
#xy_time = xy_time[1:, :]
#X = Y
###################

map_shape = [Y.shape[1], Y.shape[2]]

# Select a subset region of an AMPERE map
#Y = Y[:, 20:50, :]
#map_shape[0] = 30

# Scale the output by 100
#Y = Y * 100

# Flattern Y
Y = Y.reshape((Y.shape[0], -1))

###################
## Just for testing
#import copy
#Yn = copy.deepcopy(Y)
#Yp = copy.deepcopy(Y)
#Yn[Yn>=0] = 0
#Yp[Yp<0] = 0
#Y = np.array([Yn.sum(axis=1), Yp.sum(axis=1)]).T
#map_shape = [1,2]
###################


# Split the data into train/validation/test sets
npoints = X.shape[0]
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)
x_train = X[:train_eindex, :]
x_val = X[train_eindex:val_eindex, :]
x_test = X[val_eindex:, :]
y_train = Y[:train_eindex, :]
y_val = Y[train_eindex:val_eindex, :]
y_test = Y[val_eindex:, :]

xy_time_train = xy_time[:train_eindex, :]
xy_time_val = xy_time[train_eindex:val_eindex, :]
xy_time_test = xy_time[val_eindex:, :]


# Build a ResNet model
input_shape = X.shape[1:]
output_shape = Y.shape[1:]
if optimizer == "adam":
    optimizer=keras.optimizers.Adam(lr=learning_rate)
if optimizer == "RMSprop":
    optimizer=keras.optimizers.RMSprop(lr=learning_rate)

loss_dct = {"mean_squared_error":mse,
            "root_mean_squared_error":rmse,
            "mean_absolute_error": mae,
            "correlation_coefficient_loss":cce}
metric_dct = {"mse":mse,
              "rmse":rmse,
              "mae": mae,
              "cc":cce}

loss_func = loss_dct[loss]
metrics_func = [metric_dct[x] for x in metrics]

# Train the model
if not skip_training:
    dl_obj = CNN_MLP(input_shape, output_shape, batch_size=batch_size, n_epochs=n_epochs,
                     loss=loss_func, optimizer=optimizer,
                     hidden_layer_activation=hidden_layer_activation,
                     metrics=metrics_func, out_dir=out_dir)

    print("Training the model...")
    dl_obj.model.summary()
    fit_history = train_model(dl_obj.model, x_train, y_train, x_val, y_val,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=dl_obj.callbacks, shuffle=True)


## Plot the loss curve and the prediction accuracy
#if transfer_weights or not skip_training:
#    # Plot the training 
#    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
#    xs = np.arange(n_epochs)
#    train_loss = fit_history.history["loss"]
#    val_loss = fit_history.history["val_loss"]
#    train_acc = fit_history.history["acc"]
#    val_acc = fit_history.history["val_acc"]
#    axes[0].plot(xs, train_loss, label="train_loss") 
#    axes[0].plot(xs, val_loss, label="val_loss") 
#    axes[1].plot(xs, train_acc, label="train_acc") 
#    axes[1].plot(xs, val_acc, label="val_acc") 
#    axes[0].set_title("Training Loss and Accuracy")
#    axes[0].set_ylabel("Loss")
#    axes[1].set_ylabel("Accuracy")
#    axes[1].set_xlabel("Epoch")
#    axes[0].legend()
#    axes[1].legend()
#    fig_path = os.path.join(out_dir, "loss_acc")
#    fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

# Evaluate the model on test dataset
print("Evaluating the model...")
test_epoch = n_epochs
#test_epoch = 50    # The epoch number of the model we want to evaluate
if test_epoch < 10:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_0" + str(test_epoch) + "*hdf5"))[0]
else:
    model_name = glob.glob(os.path.join(out_dir, "weights.epoch_" + str(test_epoch) + "*hdf5"))[0]
test_model = keras.models.load_model(model_name, custom_objects=loss_dct)

# Make predictions
y_train_pred = test_model.predict(x_train, batch_size=batch_size)
y_val_pred = test_model.predict(x_val, batch_size=batch_size)
y_test_pred = test_model.predict(x_test, batch_size=batch_size)
y_pred = test_model.predict(X, batch_size=batch_size)



# Report for train data
#print("Prediction report for train input data.")
#print(classification_report(y_train_true, y_train_pred))
#
## Report for validation data
#print("Prediction report for validation input data.")
#print(classification_report(y_val_true, y_val_pred))
#
## Report for test data
#print("Prediction report for test data.")
#print(classification_report(y_test_true, y_test_pred))

if save_prediction:
    pred_dir = os.path.join(out_dir, "predicted_ampere")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)

    # Save the predicted outputs for test data, one AMPERE map per file
    for i, dtm in enumerate(xy_time_test):
        fname = hemi + "_" + dtm[0].strftime("%Y%m%d.%H%M") + ".npy"
        y_dtm = y_test_pred[i].reshape(map_shape[0], map_shape[1])
        np.save(os.path.join(pred_dir, fname), y_dtm)


