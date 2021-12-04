from Helper.InputOutput.narmax_helper_methods import *
from Model.InputOutput.narmax_model import NARMAX
from sklearn.preprocessing import MinMaxScaler
import time

CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S")
NARMAX_SAVE_LOC = 'train/servomotor/'
NARMAX_LOAD_LOC = NARMAX_SAVE_LOC + 'narmax_td8_n10'
NARMAX_LOAD_DATA = 'narmax_model_100.ph'
TRAIN_DATA = '../../../Data/ServoMotor.txt'


def main():

    """
    Main Method.

    :return: none
    """

    # Use trained narmax
    # If using already trained models, then configure
    # models save location above.
    trained_model = True

    # Training Parameters
    epochs, print_every = 10000, 100

    # NARMAX TDL Sizes for Input, Output, and Error
    delays = 8

    # Number of neurons in NARMAX model
    neurons = 10

    # Prediction horizon for training
    prediction_horizon = 100

    # Time length used for loading the NARMAX's TDL with errors.
    # Should usually be 3 times the length of the error tap-delay.
    loading_length = delays * 3

    # Data path
    path = TRAIN_DATA

    # Import data for model
    inputs, labels, scalar = import_select_data(path)

    # Setup model
    model = NARMAX(input_delay_size=delays, output_delay_size=delays, error_delay_size=delays,
                   hidden_size=neurons, input_size=1, output_size=1, activation_type='tanh')

    # Train k-Steps Ahead NARMAX model
    if not trained_model:

        print()
        print("Training k-Step Ahead NARMAX for Horizon: " + str(prediction_horizon))
        print("-------------------------------------------------------")

        # Setup Data
        input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                        prediction_horizon=prediction_horizon,
                                                        loading_length=loading_length,
                                                        overlap=False)

        # Train Model
        model, _ = train_scg(model, input_tensor, label_tensor, epochs=epochs, loading_length=loading_length,
                             print_every=print_every, plot_every=False)

        # Save Model
        folder_name = NARMAX_SAVE_LOC + "narmax_" + CURRENT_TIME
        file_name = "narmax_model_" + str(prediction_horizon) + '.ph'
        save_model(model, folder_name, file_name)

    else:
        model = load_model(model, NARMAX_LOAD_LOC, NARMAX_LOAD_DATA)

    # k-step ahead response
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=False)
    input_tensor, label_tensor = input_tensor.view(1, -1, 1), label_tensor.view(1, -1, 1)
    input_tensor, label_tensor = input_tensor[:, :10000, :], label_tensor[:, :10000, :]
    plot_response(model, input_tensor, label_tensor, loading_length)

    # Plot auto-correlation function for residual errors (prediction errors)
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=True,
                                                    loading_length=loading_length)
    input_tensor, label_tensor = input_tensor[:10000, :, :], label_tensor[:10000, :, :]
    plot_acf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4], loading_length=loading_length)
    plot_errors(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4], loading_length=loading_length)
    plot_ccf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4], loading_length=loading_length)
    plot_error_histogram(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4], loading_length=loading_length)

    # Run a skyline function for performance of plant
    plot_skyline_response(model, 10000, 200, 10, 1, -1)

    # Plot the mean square error for the k-step ahead predictions
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    loading_length=loading_length,
                                                    overlap=10)

    mae = unscaled_prediction_mae(model, input_tensor, label_tensor, scalar, loading_length)
    plt.plot(np.linspace(1, prediction_horizon, prediction_horizon),
             mae, linewidth=1.5)
    plt.title('Mean Absolute Error for NARMAX k-Step Ahead Predictor')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('k-Step Ahead Values')
    plt.show()


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

    # Import Data into DataFrame
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

    return inputs, labels, scalar


if __name__ == "__main__":
    main()
