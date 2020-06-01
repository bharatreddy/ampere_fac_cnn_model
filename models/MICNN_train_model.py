import keras
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from cnn_models import MICNN, MICNN_ResNet, train_model
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
sys.path.append("../validation/")
import overlay_ampere
import load_amp_pred_act
import model_error_stats
from create_datapoints import create_datapoints
from model_utils import load_param_data, create_results_folder
from custom_loss_functions import mse, rmse, mae, cce, mae_med_jr, rmse_med_jr

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
imf_as_input = params_dict["model_params"]["imf_as_input"]
sml_as_input = params_dict["model_params"]["sml_as_input"]
symh_as_input = params_dict["model_params"]["symh_as_input"]
f107_as_input = params_dict["model_params"]["f107_as_input"]
ampere_as_input = params_dict["model_params"]["ampere_as_input"]
month_as_input = params_dict["model_params"]["month_as_input"]
n_resnet_units_imf = params_dict["model_params"]["n_resnet_units_imf"]
n_resnet_units_ampere = params_dict["model_params"]["n_resnet_units_ampere"]
n_filters_imf = params_dict["model_params"]["n_filters_imf"]
n_filters_ampere = params_dict["model_params"]["n_filters_ampere"]
kernel_size_imf = params_dict["model_params"]["kernel_size_imf"]
kernel_size_ampere = params_dict["model_params"]["kernel_size_ampere"]
n_layers_per_resnet_unit = params_dict["model_params"]["n_layers_per_resnet_unit"]

input_params = params_dict["general_params"]["input_params"]
param_col_dict = {"Bx":0, "By":1, "Bz":2, "Vx":3, "Np":4, "au":5, "al":6}
input_cols = [param_col_dict[x] for x in input_params]

optimizer = params_dict["model_params"]["optimizer"]
learning_rate = params_dict["model_params"]["learning_rate"]
loss = params_dict["model_params"]["loss"]
hidden_layer_activation = params_dict["model_params"]["hidden_layer_activation"]
save_prediction = params_dict["model_params"]["save_prediction"]
plot_prediction = params_dict["model_params"]["plot_prediction"]
plot_time_step = params_dict["model_params"]["plot_time_step"]
make_err_analys = params_dict["model_params"]["make_err_analys"]

n_epochs = params_dict["model_params"]["n_epochs"]
batch_size = params_dict["model_params"]["batch_size"]
metrics = params_dict["model_params"]["metrics"]
train_size = params_dict["general_params"]["train_size"]
val_size = params_dict["general_params"]["val_size"]
test_size = params_dict["general_params"]["test_size"]

skip_training = False
#skip_training = True

# Construct data points
print("loading data points...")
dataset_fname = hemi + "_" + start_datetime.strftime("%Y%m%d") +\
                "_" + end_datetime.strftime("%Y%m%d") + ".npz"
dataset_file = os.path.join(dataset_dir, dataset_fname)
datapoints = np.load(dataset_file)

# Select certain columns
X_imf = datapoints["X_OMN"]
X_imf = X_imf[:, :, input_cols]
X_sml = datapoints["X_SML"]
X_symh = datapoints["X_SYM"]
X_f107 = datapoints["X_F107"]
Y = datapoints["Y"]
Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2], -1))  # This step may not be neccessary but just in case.
xy_time = datapoints["time"]

if (not imf_as_input) and (not ampere_as_input):
    print("At least one of imf_as_input and ampere_as_input should be True")
if imf_as_input:
    if sml_as_input:
        X_imf = np.concatenate([X_imf, X_sml], axis=2)
    if symh_as_input:
        X_imf = np.concatenate([X_imf, X_symh], axis=2)
    if f107_as_input:
        X_imf = np.concatenate([X_imf, X_f107], axis=2)

    # Add Month as input feature
    if month_as_input:
        del_min = np.array([np.timedelta64(i-X_imf.shape[1], "m") for i in range(1, X_imf.shape[1]+1)])
        del_mins = np.tile(del_min, (X_imf.shape[0], 1))
        dtms = np.tile(xy_time.reshape((-1, 1)), (1, X_imf.shape[1]))
        #dtms = dtms.astype("datetime64[m]") + del_mins
        months = (dtms.astype("datetime64[M]") - dtms.astype("datetime64[Y]")).astype("timedelta64[M]").astype(int) + 1
        months = months.reshape((months.shape[0], X_imf.shape[1], 1))
        # Encode months using sine and cosine functions
        months_sine = np.sin(2*np.pi/12 * months)
        months_cosine = np.cos(2*np.pi/12 * months)
        X_imf = np.concatenate([X_imf, months_sine, months_cosine], axis=2)
else:
    print("NOTE: Since you set imf_as_input as False, switchs such as sml_as_input, " +\
          "symh_as_input, f107_as_input, and f107_as_input are all ignored.")

# Limit SW & IMF time history 
X_imf = X_imf[:, -omn_history:, :]

# Limit the MLAT region in AMPERE map
discarded_mlat = 0  # unit in degrees
# Fill the discarded elements with zeros and pad it latter afte prediction
if discarded_mlat:
    arr_0 = np.zeros((discarded_mlat, Y.shape[2])) + 0.0001
    Y = Y[:,discarded_mlat:, :]

###################
# Prepare the inputs 
# Set the # previous AMPERE maps.
ampere_hist = 1
if not ampere_as_input:
    ampere_hist = 0
rw = X_imf.shape[0] - ampere_hist

# Prepare AMPERE data for input
if ampere_as_input:
    X_prev= []
    for i in range(ampere_hist):
        X_prev.append(Y[i:-(ampere_hist-i), :])
    if len(X_prev)>1:
        X_ampere = np.concatenate(X_prev, axis=3)
    else:
        X_ampere = X_prev[0]

    # Normalize AMPERE input
    X_ampere = (X_ampere - X_ampere.mean()) / X_ampere.std()

# Prepare SW&IMF for input
if imf_as_input:
    X_imf = X_imf[ampere_hist:, :]

# Combine the inputs
if ampere_as_input & imf_as_input:
    X = [X_imf, X_ampere]
elif ampere_as_input:
    X = [X_ampere]
elif imf_as_input:
    X = [X_imf]

# Prepare output data
if hemi != "both":
    Y = Y[ampere_hist:, :]
xy_time = xy_time[ampere_hist:, :]

####################
## Just for testing the identify mapping
#X[1] = Y
#X_ampere = Y
####################

output_map_shape = [Y.shape[1], Y.shape[2]]

# Flatten Y
Y = Y.reshape((Y.shape[0], -1, Y.shape[-1]))

# Split the data into train/validation/test sets
npoints = Y.shape[0]
train_eindex = int(npoints * train_size)
val_eindex = train_eindex + int(npoints * val_size)

x_train, x_val, x_test = [], [], []
for xi in X:
    x_train.append(xi[:train_eindex, :])
    x_val.append(xi[train_eindex:val_eindex, :])
    x_test.append(xi[val_eindex:, :])

y_train, y_val, y_test = [], [], []
for k in range(Y.shape[-1]):
    y_train.append(Y[:train_eindex, :, k])
    y_val.append(Y[train_eindex:val_eindex, :, k])
    y_test.append(Y[val_eindex:, :, k])

xy_time_train = xy_time[:train_eindex, :]
xy_time_val = xy_time[train_eindex:val_eindex, :]
xy_time_test = xy_time[val_eindex:, :]

print( min(xy_time), max(xy_time) )
print( min(xy_time_test), max(xy_time_test) )


# Build a ResNet model
input_shape_imf = X_imf.shape[1:]
if ampere_as_input:
    input_shape_ampere = X_ampere.shape[1:]
else:
    input_shape_ampere = None 

output_shape = Y.shape[1:]
if optimizer == "adam":
    optimizer=keras.optimizers.Adam(lr=learning_rate)
if optimizer == "RMSprop":
    optimizer=keras.optimizers.RMSprop(lr=learning_rate)

loss_dct = {"mse":mse,
            "rmse":rmse,
            "mae": mae,
            "cce":cce,
	    "mae_med_jr":mae_med_jr,
            "rmse_med_jr":rmse_med_jr}
metric_dct = loss_dct
loss_func = loss_dct[loss]
metrics_func = [metric_dct[x] for x in metrics]

# Train the model
if not skip_training:
    dl_obj = MICNN_ResNet(input_shape_imf, input_shape_ampere, output_shape, batch_size=batch_size,
                 n_epochs=n_epochs, loss=loss_func, optimizer=optimizer,
                 hidden_layer_activation=hidden_layer_activation,
                 ampere_as_input=ampere_as_input,
                 imf_as_input=imf_as_input,
                 metrics=metrics_func,
                 n_resnet_units_imf=n_resnet_units_imf,
                 n_resnet_units_ampere=n_resnet_units_ampere,
                 n_filters_imf=n_filters_imf,
                 n_filters_ampere=n_filters_ampere,
                 kernel_size_imf=kernel_size_imf,
                 kernel_size_ampere=kernel_size_ampere,
                 n_layers_per_resnet_unit=n_layers_per_resnet_unit,
                 hemi=hemi,
                 out_dir=out_dir)

    print("Training the model...")
    dl_obj.model.summary()
    fit_history = train_model(dl_obj.model, x_train, y_train, x_val, y_val,
                              batch_size=batch_size, n_epochs=n_epochs,
                              callbacks=dl_obj.callbacks, shuffle=True)

if hemi == "both":
    hemi_list = ["north", "south"]
else:
    hemi_list = [hemi]
# Plot the loss curve and the evaluation metrics
if not skip_training:
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    xs = np.arange(n_epochs)

    # Plot the loss curves
    train_loss = fit_history.history["loss"]
    val_loss = fit_history.history["val_loss"]
    axes[0].plot(xs, train_loss, label="train_loss") 
    axes[0].plot(xs, val_loss, label="val_loss") 
    axes[0].set_title("Loss and Error Curves")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot the error curves
    train_errs, val_errs = [], []
    for m in metrics:
        for hm in hemi_list:
            if len(hemi_list) == 1:
                txt = "" 
            else:
                txt = hm + "_"
            train_err = fit_history.history[txt + m]
            val_err = fit_history.history["val_" + txt + m]
            axes[1].plot(xs, train_err, label="train_" + txt + m) 
            axes[1].plot(xs, val_err, label="val_" + txt + m) 
    axes[1].set_ylabel("Errors")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig_path = os.path.join(out_dir, "loss_and_error_curves")
    fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")  

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

if save_prediction:
    print("output model directory-->" + out_dir)
    pred_dir = os.path.join(out_dir, "predicted_ampere/")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)

    # Save the predicted outputs for test data, one AMPERE map per file
    for i, dtm in enumerate(xy_time_test):
        for h, hm in enumerate(hemi_list):
            fname = hm + "_" + dtm[0].strftime("%Y%m%d.%H%M") + ".npy"
            y_dtm = y_test_pred[h][i].reshape(output_map_shape[0], output_map_shape[1])
            if discarded_mlat:
                y_dtm = np.vstack([arr_0, y_dtm])
            np.save(os.path.join(pred_dir, fname), y_dtm)
    # if we choose to make plots
    # Note we can plot predictions only if we save them!
    if plot_prediction:
        read_north = False 
        read_south = False 
        if hemi == "both":
            read_north = True
            read_south = True
        if hemi == "north":
            read_north = True
        if hemi == "south":
            read_south = True
        print("plotting data!")
        plot_dir = os.path.join(out_dir, "plots/")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        # location of actual ampere files
        act_dir = '../data/ampere/'
        plot_sdate = min(xy_time_test.reshape(xy_time_test.shape[0]))
        plot_edate = max(xy_time_test.reshape(xy_time_test.shape[0]))
        plot_dates = pd.date_range(start=plot_sdate, end=plot_edate,
                                       freq=str(plot_time_step)+"Min")
        filter_jr_magns = {"actual_jr": 0.1, "pred_jr": 0.05}
        out_format = "png"
        # setup obj
        amp_obj = load_amp_pred_act.LoadAmp(act_dir, pred_dir, 
                                            read_north=read_north, 
                                            read_south=read_south, 
                                            date_range=[plot_sdate, plot_edate])
        amp_data_df = amp_obj.create_amp_dataframes()
        amp_cmpr_file = os.path.join(out_dir, "model_act_cmpr.csv")
        amp_data_df.to_csv(amp_cmpr_file)
        # first check if we wanted to perform analysis
        if make_err_analys:
            print("Generating error stats file and plots")
            amp_err_obj = model_error_stats.ErrorStats(plot_dir,\
                         pred_df=amp_data_df, read_predictions_from_disk=False)
            err_data = amp_err_obj.get_error_mlat_mlt()
            # save err_data
            err_file = os.path.join(out_dir, "model_error_stats.csv")
            err_data.to_csv(err_file)
            # generate pcolormesh plot for different params
            plot_param_list = ['abs_jr_err_median', 'rel_jr_err_median',\
			       'jr_sign_err_median', 'abs_jr_err_std', 'rel_jr_err_std',\
			       'jr_sign_err_std', 'perc_right']
            for _par in plot_param_list:
                fig1 = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(1, 1, 1, projection='polar')
                amp_err_obj.poly_plot( fig1, ax1, err_data, plot_param=_par )
        # setup obj for plotting
        plr_plot_obj = overlay_ampere.OverlayAMP(amp_data_df, plot_dir)
        for plot_date in plot_dates:
            for hm in hemi_list:
                plr_plot_obj.act_pred_cmpr_poly_plot(plot_date,\
                                     filter_act_jr_magn=filter_jr_magns["actual_jr"],
                                     filter_pred_jr_magn=filter_jr_magns["pred_jr"],\
                                     plot_hemi=hm, out_format=out_format)

