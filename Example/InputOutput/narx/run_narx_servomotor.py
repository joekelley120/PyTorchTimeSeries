from Helper.InputOutput.narx_helper_methods import *
from Model.InputOutput.narx_model import NARX
from sklearn.preprocessing import MinMaxScaler


CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S")
NARX_SAVE_LOC = 'train/servomotor/'
NARX_LOAD_LOC = NARX_SAVE_LOC + 'narx_td8_n10'
NARX_LOAD_DATA = 'narx_model_100.ph'
TRAIN_DATA = '../../../Data/ServoMotor.txt'


def main():

    """
    Main Method.

    :return: none
    """

    # Use trained narx
    # If using already trained models, then configure
    # models save location above.
    trained_model = False

    # Training Parameters
    epochs, print_every = 10000, 100

    # NARX TDL Sizes for Input and Output
    delays = 15

    # Number of neurons in NARX model
    neurons = 10

    # Prediction horizon for training
    prediction_horizon = 100

    # Time length used for loading the NARX's TDL with errors.
    # Should usually be 3 times the length of the error tap-delay.
    loading_length = delays * 3

    # Data path
    path = TRAIN_DATA

    # Import data for model
    inputs, labels, scalar = import_select_data(path)

    # Setup model
    model = NARX(input_delay_size=delays, output_delay_size=delays, hidden_size=neurons,
                 input_size=1, output_size=1, activation_type='tanh')

    # Train k-Steps Ahead NARX model
    if not trained_model:

        print()
        print("Training k-Step Ahead NARX for Horizon: " + str(prediction_horizon))
        print("-------------------------------------------------------")

        # Setup Data
        input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                        prediction_horizon=prediction_horizon,
                                                        overlap=False)

        # Train Model
        model, _ = train_scg(model, input_tensor, label_tensor, epochs=epochs,
                             print_every=print_every, plot_every=False)

        # Save Model
        folder_name = NARX_SAVE_LOC + "narx_" + CURRENT_TIME
        file_name = "narx_model_" + str(prediction_horizon) + '.ph'
        save_model(model, folder_name, file_name)

    else:
        model = load_model(model, NARX_LOAD_LOC, NARX_LOAD_DATA)

    # k-step ahead response
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=False)
    input_tensor, label_tensor = input_tensor.view(1, -1, 1), label_tensor.view(1, -1, 1)
    input_tensor, label_tensor = input_tensor[:, :5000, :], label_tensor[:, :5000, :]
    plot_response(model, input_tensor, label_tensor)

    # Plot auto-correlation function for residual errors (prediction errors)
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=True)
    input_tensor, label_tensor = input_tensor[:5000, :, :], label_tensor[:5000, :, :]
    plot_acf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_errors(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_ccf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_error_histogram(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])

    # Run a skyline function for performance of plant
    plot_skyline_response(model, 10000, 200, 10, 1, -1)

    # Plot the mean square error for the k-step ahead predictions
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    loading_length=loading_length,
                                                    overlap=False)

    mae = unscaled_prediction_mae(model, input_tensor, label_tensor, scalar, loading_length=loading_length)
    plt.plot(np.linspace(1, prediction_horizon, prediction_horizon),
             mae, linewidth=1.5)
    plt.title('Mean Absolute Error for NARX k-Step Ahead Predictor')
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
    size = floor(len(data))
    inputs = np.zeros((size, 1))
    labels = np.zeros((size, 1))
    for i in range(size):
        inputs[i, 0] = data_scaled[i, 0]
        labels[i, 0] = data_scaled[i, 1]

    return inputs, labels, scalar


if __name__ == "__main__":
    main()
