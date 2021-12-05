from Helper.helper_methods import *
import Helper.Controller.mrac_helper_methods as mrac_helper
import Helper.InputOutput.narx_helper_methods as narx_helper
from Model.Controller.mrac_model import Controller, ControllerConfiguration
from Model.InputOutput.narx_model import NARX
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import time


CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S")
PLANT_SAVE_LOC = 'train/maglev/'
CONTROLLER_SAVE_LOC = 'train/maglev/'
PLANT_LOAD_LOC = PLANT_SAVE_LOC + 'mrac_ptd5_ctd5_n3'
CONTROLLER_LOAD_LOC = CONTROLLER_SAVE_LOC + 'mrac_ptd6_ctd6_n5'
PLANT_LOAD_DATA = 'plant_model_100.ph'
CONTROLLER_LOAD_DATA = 'controller_model_1200.ph'
TRAIN_DATA = '../../../Data/MagLev.txt'


def main():

    """
    Main Method.

    :return: none
    """

    # Use trained plant or controller models
    # If using already trained models, then configure
    # models save location above.
    # The plant model needs to be trained before
    # training or running the controller.
    trained_plant, trained_controller = False, False

    # Load and retrain mrac controller model
    retrain_controller_from_last_save = False

    # Training Parameters
    epochs, print_every = 10000, 10

    # NARX TDL Sizes for Input and Output
    narx_delay = 5

    # NARX Zero Input Delay
    narx_zero_input_delay = True

    # MRAC TDL Sizes for Input, Reference, and Output
    mrac_delay = 5

    # Number of neurons in NARX model
    narx_neurons = 3

    # Number of neurons in MRAC model
    mrac_neurons = 3

    # Prediction horizon for training
    prediction_horizon_plant = 1200
    prediction_horizon_controller = 1200

    # Data path
    path = TRAIN_DATA

    # Import data for model
    inputs, labels = import_select_data(path)

    # Setup Plant models
    plant = NARX(narx_delay, narx_delay, narx_neurons, 1, 1, zero_input_delay=narx_zero_input_delay)

    # Train plant model
    if not trained_plant:
        plant = train_plant(plant, inputs, labels,
                            prediction_horizon=prediction_horizon_plant,
                            epochs=epochs, print_every=print_every,
                            overlap=25, device=None, number_batches=1)
    else:
        plant = load_model(plant, PLANT_LOAD_LOC, PLANT_LOAD_DATA)

    # Plot plant response
    input_tensor, label_tensor = narx_helper.init_training_data(plant, inputs, labels,
                                                                prediction_horizon=prediction_horizon_plant,
                                                                overlap=False)
    input_tensor, label_tensor = input_tensor.view(1, -1, 1), label_tensor.view(1, -1, 1)
    input_tensor, label_tensor = input_tensor[:, 1000:5000, :], label_tensor[:, 1000:5000, :]
    narx_helper.plot_response(plant, input_tensor, label_tensor)

    # Run a skyline function for performance of plant
    narx_helper.plot_skyline_response(plant, 4000, 100, 20, 1, -1)

    # Setup Controller models
    plus_one = 1 if narx_zero_input_delay is True else 0
    iw = plant.cell.fully[0].w[:, :(plus_one + narx_delay)]
    ow = plant.cell.fully[0].w[:, (plus_one + narx_delay):]
    b1 = plant.cell.fully[0].b
    lw = plant.cell.lw
    b2 = plant.cell.b2
    controller_setup = ControllerConfiguration(mrac_delay, mrac_delay, mrac_delay, mrac_neurons,
                                               narx_delay, narx_delay, narx_neurons,
                                               iw, ow, b1, lw, b2)
    controller = Controller(controller_setup)

    # Train controller model
    if not trained_controller:

        # Load in train controller to continue training
        if retrain_controller_from_last_save:
            controller = load_model(controller, CONTROLLER_LOAD_LOC, CONTROLLER_LOAD_DATA)

        # Create training data for controller.
        # Using first order response as the desired response from the
        # nonlinear controller. Can be modified by the designer.
        # h = 0.02 sec (or 50 Hz)
        points = 50000
        max_width, min_width = 1000, 50
        max_height, min_height = 0.4, -0.4
        input_tensor = skyline(points, max_width, min_width, max_height, min_height)
        label_tensor = first_order_response(input_tensor, time_constant=1.0, h=0.02)
        inputs = input_tensor.view(-1, 1).data.numpy()
        labels = label_tensor.view(-1, 1).data.numpy()

        # Plot the Desired response for the nonlinear system.
        plt.plot(inputs)
        plt.plot(labels)
        plt.show()

        # Begin Training
        controller = train_controller(controller, inputs, labels,
                                      prediction_horizon=prediction_horizon_controller,
                                      epochs=epochs, print_every=print_every,
                                      overlap=25, device=None, number_batches=1)

    else:
        controller = load_model(controller, CONTROLLER_LOAD_LOC, CONTROLLER_LOAD_DATA)

    # Run a skyline function for performance
    mrac_helper.plot_skyline_response(controller, 20000, 800, 10, 0.4, -0.4, 1.0, 0.02)


def import_data(filename):

    """
    Load Data from CSV File.

    :param filename: File Name and Path
    :return: Loaded Data
    """

    return np.genfromtxt(filename, delimiter=',')


def import_select_data(filename):

    """
    Load Selected Data and Convert to Numpy Arrays.

    :param filename: File Name and Path
    :return: Input and Label Data
    """

    # Import MagLev Data into DataFrame
    data = import_data(filename)

    # Scale Input and Output Data
    scalar = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scalar.fit_transform(data)

    # Append to Array
    size = floor(len(data) / 2)
    inputs = np.zeros((size, 1))
    labels = np.zeros((size, 1))
    for i in range(size):
        inputs[i, 0] = data_scaled[i, 0]
        labels[i, 0] = data_scaled[i, 1]

    return inputs, labels


def train_plant(model, inputs, labels, prediction_horizon=1, epochs=5000,
                print_every=500, overlap=False, device=None, number_batches=1):

    """
    Train plant model.

    :param model: model
    :param inputs: inputs
    :param labels: labels
    :param prediction_horizon: training prediction horizon
    :param epochs: training epochs per prediction horizon
    :param print_every: print to console very number of epochs.
    :param overlap: data will be sorted with overlap in time
    :param device: device type. Default is None, which means we will use 'gpu' if available
    :param number_batches: number of training batches
    :return: model
    """

    print()
    print("Training Plant for Horizon: " + str(prediction_horizon))
    print("-------------------------------------------------------")

    # Setup Data
    input_tensor, label_tensor = narx_helper.init_training_data(model, inputs, labels,
                                                                prediction_horizon=prediction_horizon,
                                                                overlap=overlap)

    # Train Model
    model, losses = narx_helper.train_scg(model, input_tensor, label_tensor, epochs=epochs,
                                          print_every=print_every, plot_every=False,
                                          device=device, number_batches=number_batches)

    # Save Model
    folder_name = PLANT_SAVE_LOC + "mrac_" + CURRENT_TIME
    file_name = "plant_model_" + str(prediction_horizon) + '.ph'
    save_model(model, folder_name, file_name)

    return model


def train_controller(model, inputs, labels, prediction_horizon=1, epochs=5000,
                     print_every=500, overlap=False, device=None, number_batches=1):

    """
    Train controller model.

    :param model: model
    :param inputs: inputs
    :param labels: labels
    :param prediction_horizon: training prediction horizon
    :param epochs: training epochs per prediction horizon
    :param print_every: print to console very number of epochs.
    :param overlap: data will be sorted with overlap in time
    :param device: device type. Default is None, which means we will use 'gpu' if available
    :param number_batches: number of training batches
    :return: model
    """

    print()
    print("Training Controller for Horizon: " + str(prediction_horizon))
    print("-------------------------------------------------------")

    # Setup Data
    input_tensor, label_tensor = mrac_helper.init_training_data(inputs, labels,
                                                                prediction_horizon=prediction_horizon,
                                                                overlap=overlap)

    # Train Model
    model, losses = mrac_helper.train_scg(model, input_tensor, label_tensor, epochs=epochs,
                                          print_every=print_every, plot_every=False,
                                          number_batches=number_batches, device=device)

    # Save Model
    folder_name = CONTROLLER_SAVE_LOC + "mrac_" + CURRENT_TIME
    file_name = "controller_model_" + str(prediction_horizon) + ".ph"
    save_model(model, folder_name, file_name)

    return model


if __name__ == "__main__":
    main()
