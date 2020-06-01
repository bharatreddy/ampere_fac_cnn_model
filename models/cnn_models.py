from keras.layers import Input, Conv1D, Conv2D, Dense, Flatten, Add
from keras.layers import normalization, Activation, pooling, concatenate
from keras.models import Model 
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers.core import Dropout
import os


class CNN_MLP:

    ''' A Convolutional Neural Network (CNN) followed by a Multi-layer Perceptrons (MLP) '''
    def __init__(self, input_shape, output_shape, batch_size=32, n_epochs=100,
                 loss="mse", optimizer="adam", metrics=["mse"],
                 hidden_layer_activation="relu",
                 out_dir="./trained_models/CNN_MLP/"):

        # Add class attributes
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir
        self.hidden_layer_activation = hidden_layer_activation

        # Creat a FCNN model
        self.model = self.creat_model()

    def creat_model(self):

        # Input layer
        input_layer = Input(self.input_shape)

        conv_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding="valid")(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation=self.hidden_layer_activation)(conv_layer)
        # Max pooling layer
        #conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)

        conv_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding="valid")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation=self.hidden_layer_activation)(conv_layer)

        # Max pooling layer
        conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)

        # Flatten 2D data into 1D
        fc_layer = Flatten()(conv_layer)
        #fc_layer = Dropout(0.5, seed=100)(fc_layer)

        # Add Dense layer 
        fc_layer = Dense(1000, activation=self.hidden_layer_activation)(fc_layer)
        #fc_layer = Dropout(0.2, seed=100)(fc_layer)

#        ####################
#        # Add Dense layer 
#        fc_layer = Dense(200, activation=self.hidden_layer_activation)(fc_layer)
#        #fc_layer = Dropout(0.2, seed=100)(fc_layer)
#
#        fc_layer = Dense(200, activation=self.hidden_layer_activation)(fc_layer)
#        #fc_layer = Dropout(0.2, seed=100)(fc_layer)
        ####################

        ## Add Dense layer 
        fc_layer = Dense(300, activation=self.hidden_layer_activation)(fc_layer)

        # Output layer
        num_output_nodes = self.output_shape[0]
        output_layer = Dense(num_output_nodes, activation="linear")(fc_layer)

        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layer)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

class MICNN:

    ''' Convolutional Neural Network (CNN) that accepts two different input data.
        MICNN stands for Multi-Input CNN.
    '''
    def __init__(self, input_shape_imf, input_shape_ampere, output_shape, 
                 batch_size=32, n_epochs=100, loss="mse",
                 optimizer="adam", metrics=["mse"],
                 hidden_layer_activation="relu",
                 ampere_as_input=False,
                 imf_as_input=True,
                 out_dir="./trained_models/CNN_MLP/"):

        # Add class attributes
        self.input_shape_imf = input_shape_imf
        self.input_shape_ampere = input_shape_ampere
        self.output_shape = output_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir
        self.hidden_layer_activation = hidden_layer_activation
        self.imf_as_input = imf_as_input
        self.ampere_as_input = ampere_as_input

        # Creat a FCNN model
        self.model = self.creat_model()

    def creat_model(self):

        num_output_nodes = self.output_shape[0]

        if self.imf_as_input:
        # Input layer for SW &IMF data
            input_layer_imf = Input(self.input_shape_imf)

            conv_layer_imf = Conv1D(filters=32, kernel_size=3, strides=2, padding="valid")(input_layer_imf)
            conv_layer_imf = normalization.BatchNormalization()(conv_layer_imf)
            conv_layer_imf = Activation(activation=self.hidden_layer_activation)(conv_layer_imf)

            conv_layer_imf = Conv1D(filters=32, kernel_size=3, strides=1, padding="valid")(conv_layer_imf)
            conv_layer_imf = normalization.BatchNormalization()(conv_layer_imf)
            conv_layer_imf = Activation(activation=self.hidden_layer_activation)(conv_layer_imf)

            ## Max pooling layer
            #conv_layer_imf = pooling.MaxPooling1D(pool_size=2)(conv_layer_imf)

            # Flatten 2D data into 1D
            fc_layer_imf = Flatten()(conv_layer_imf)

            # Add a dense layer whose # nodes are the same as the AMPERE output
            fc_layer_imf = Dense(num_output_nodes, activation="linear")(fc_layer_imf)

        if self.ampere_as_input:
            # Input layer for AMPERE data 
            input_layer_ampere = Input(self.input_shape_ampere)

            conv_layer_ampere = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(input_layer_ampere)
            conv_layer_ampere = normalization.BatchNormalization()(conv_layer_ampere)
            conv_layer_ampere = Activation(activation=self.hidden_layer_activation)(conv_layer_ampere)
            
            for i in range(5):
                conv_layer_ampere = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(conv_layer_ampere)
                conv_layer_ampere = normalization.BatchNormalization()(conv_layer_ampere)
                conv_layer_ampere = Activation(activation=self.hidden_layer_activation)(conv_layer_ampere)

            conv_layer_ampere = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(conv_layer_ampere)
            conv_layer_ampere = normalization.BatchNormalization()(conv_layer_ampere)
            #conv_layer_ampere = Activation(activation=self.hidden_layer_activation)(conv_layer_ampere)
            conv_layer_ampere = Activation(activation="linear")(conv_layer_ampere)

            # Flatten 2D data into 1D
            fc_layer_ampere = Flatten()(conv_layer_ampere)
            #fc_layer_ampere = Dropout(0.5, seed=100)(fc_layer_ampere)

        # Construct Fusion & Inputs according to the types of input datasets
        if self.ampere_as_input & self.imf_as_input:
            # Combine AMPERE and SW&IMF features
            #fc_layer = concatenate([fc_layer_imf, fc_layer_ampere])
            fc_layer = Add()([fc_layer_imf, fc_layer_ampere])
            inputs = [input_layer_imf, input_layer_ampere] 
        elif self.ampere_as_input:
            fc_layer = fc_layer_ampere
            inputs = [input_layer_ampere]
        elif self.imf_as_input:
            fc_layer = fc_layer_imf
            inputs = [input_layer_imf]
        else:
            print("Either ampere_as_input or imf_as_input should be set to True.")
            return None

#        # Add Dense layers
#        fc_layer = Dense(500, activation=self.hidden_layer_activation)(fc_layer)
#        fc_layer = Dropout(0.2, seed=100)(fc_layer)
#
#        fc_layer = Dense(200, activation=self.hidden_layer_activation)(fc_layer)
#        fc_layer = Dropout(0.1, seed=100)(fc_layer)

#        fc_layer = Dense(num_output_nodes, activation="linear")(fc_layer)

        # Output layer
        output_layer = fc_layer

        # Put all the model components together
        model = Model(inputs=inputs, outputs=output_layer)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

class MICNN_ResNet:

    ''' Convolutional Neural Network (CNN) that accepts two different input data.
        MICNN_ResNet stands for Multi-Input CNN with ResNet Units.
    '''
    def __init__(self, input_shape_imf, input_shape_ampere, output_shape, 
                 batch_size=32, n_epochs=100, loss="mse",
                 optimizer="adam", metrics=["mse"],
                 hidden_layer_activation="relu",
                 ampere_as_input=False,
                 imf_as_input=True,
                 n_resnet_units_imf=3,
                 n_resnet_units_ampere=3,
                 n_filters_imf=64,
                 n_filters_ampere=64,
                 n_layers_per_resnet_unit=2,
                 kernel_size_imf=3,
                 kernel_size_ampere=3,
                 hemi = "north",
                 out_dir="./trained_models/MICNN_ResNet/"):

        # Add class attributes
        self.input_shape_imf = input_shape_imf
        self.input_shape_ampere = input_shape_ampere
        self.output_shape = output_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_resnet_units_imf = n_resnet_units_imf
        self.n_resnet_units_ampere = n_resnet_units_ampere
        self.n_filters_imf = n_filters_imf
        self.n_filters_ampere = n_filters_ampere
        self.kernel_size_imf = kernel_size_imf
        self.kernel_size_ampere = kernel_size_ampere
        self.n_layers_per_resnet_unit = n_layers_per_resnet_unit
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir
        self.hidden_layer_activation = hidden_layer_activation
        self.imf_as_input = imf_as_input
        self.ampere_as_input = ampere_as_input
        self.hemi = hemi

        # Creat a FCNN model
        self.model = self.creat_model()

    def __create_resnet_unit(self, resnet_input_layer, n_filters=64, n_layers_per_resnet_unit=2,
                             kernel_sizes=[3, 3], conv_type = "1D",
                             is_first_resnet_unit=True):

        from keras.layers import add

        tmp_layer = resnet_input_layer
        for i in range(n_layers_per_resnet_unit):
            if conv_type == "1D":
                conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_sizes[i], padding='same')(tmp_layer)
            if conv_type == "2D":
                conv_layer = Conv2D(filters=n_filters, kernel_size=kernel_sizes[i], padding='same')(tmp_layer)
            conv_layer = normalization.BatchNormalization()(conv_layer)
            if i < n_layers_per_resnet_unit-1:
                conv_layer = Activation(self.hidden_layer_activation)(conv_layer)
                #conv_layer = Dropout(0.2, seed=100)(conv_layer)
            tmp_layer = conv_layer

        # expand the first resnet channels for the sum 
        if is_first_resnet_unit:
            if conv_type == "1D":
                reslink = Conv1D(filters=n_filters, kernel_size=1, padding='same')(resnet_input_layer)
            if conv_type == "2D":
                reslink = Conv2D(filters=n_filters, kernel_size=1, padding='same')(resnet_input_layer)
        else:
            reslink = resnet_input_layer
        reslink = normalization.BatchNormalization()(reslink)

        output_layer = add([reslink, tmp_layer])
        output_layer = Activation(self.hidden_layer_activation)(output_layer)

        return output_layer

    def creat_model(self):

        num_output_nodes = self.output_shape[0]   # deternied by the AMPERE output map

        # Conv. layer for SW &IMF data
        if self.imf_as_input:
            input_layer_imf = Input(self.input_shape_imf)

            conv_layer_imf = input_layer_imf
            # ResNet Units for IMF
            resnet_unit_input = conv_layer_imf
            for i in range(self.n_resnet_units_imf):
                if i == 0:
                    is_first_resnet_unit=True
                else:
                    is_first_resnet_unit=False
                resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=self.n_filters_imf,
                                                               n_layers_per_resnet_unit=self.n_layers_per_resnet_unit,
                                                               conv_type="1D",
                                                               kernel_sizes=[self.kernel_size_imf]*self.n_layers_per_resnet_unit,
                                                               is_first_resnet_unit=is_first_resnet_unit)
                resnet_unit_input = resnet_unit_output


            # Flatten 2D data into 1D
            fc_layer_imf = Flatten()(resnet_unit_output)

            # Add a dense layer whose # nodes are the same as the AMPERE output
            if self.hemi != "both":
                fc_layer_imf = Dense(num_output_nodes, activation="linear", name=self.hemi)(fc_layer_imf)
            else:
                fc_layer_imf_n = Dense(num_output_nodes, activation="linear", name="north")(fc_layer_imf)
                fc_layer_imf_s = Dense(num_output_nodes, activation="linear", name="south")(fc_layer_imf)

        # Conv layer for AMPERE data 
        if self.ampere_as_input:
            input_layer_ampere = Input(self.input_shape_ampere)

            # ResNet Units
            resnet_unit_input = input_layer_ampere
            for i in range(self.n_resnet_units_ampere):
                if i == 0:
                    is_first_resnet_unit=True
                else:
                    is_first_resnet_unit=False
                resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=self.n_filters_ampere,
                                                               n_layers_per_resnet_unit=self.n_layers_per_resnet_unit,
                                                               conv_type = "2D",
                                                               kernel_sizes=[self.kernel_size_ampere]*self.n_layers_per_resnet_unit,
                                                               is_first_resnet_unit=is_first_resnet_unit)
                resnet_unit_input = resnet_unit_output
    
            conv_layer_ampere = Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(resnet_unit_output)
            conv_layer_ampere = normalization.BatchNormalization()(conv_layer_ampere)
            conv_layer_ampere = Activation(activation="linear")(conv_layer_ampere)

            # Flatten 2D data into 1D
            fc_layer_ampere = Flatten()(conv_layer_ampere)
            #fc_layer_ampere = Dropout(0.5, seed=100)(fc_layer_ampere)

        # Construct Fusion & Inputs according to the types of input datasets
        if self.ampere_as_input & self.imf_as_input:
            # Combine AMPERE and SW&IMF features
            #fc_layer = concatenate([fc_layer_imf, fc_layer_ampere])
            fc_layer = Add()([fc_layer_imf, fc_layer_ampere])
            inputs = [input_layer_imf, input_layer_ampere] 
        elif self.ampere_as_input:
            outupts = [fc_layer_ampere]
            inputs = [input_layer_ampere]
        elif self.imf_as_input:
            if self.hemi != "both":
                outputs = [fc_layer_imf]
            else:
                outputs = [fc_layer_imf_n, fc_layer_imf_s]
            inputs = [input_layer_imf]
        else:
            print("Either ampere_as_input or imf_as_input should be set to True.")
            return None

        # Put all the model components together
        model = Model(inputs=inputs, outputs=outputs)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

def train_model(model, x_train, y_train, x_val, y_val,
                batch_size=32, n_epochs=10,
                callbacks=None, shuffle=True):

    from keras.backend import clear_session
    import datetime as dt

    # Train the model
    stime = dt.datetime.now() 
    fit_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, 
                            validation_data=(x_val, y_val),
                            callbacks=callbacks, shuffle=shuffle)
    etime = dt.datetime.now() 
    training_time = (etime - stime).total_seconds()/60.    # minutes
    print("Training time is {tm} minutes".format(tm=training_time))

    # Test the model on evaluation data
    clear_session()

    return fit_history

