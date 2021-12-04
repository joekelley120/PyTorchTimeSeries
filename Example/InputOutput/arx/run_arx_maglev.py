from Helper.InputOutput.arx_helper_methods import *
from Model.InputOutput.arx_model import ARX
from sklearn.preprocessing import MinMaxScaler


CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S")
ARX_SAVE_LOC = 'train/maglev/'
ARX_LOAD_LOC = ARX_SAVE_LOC + 'arx_td2'
ARX_LOAD_DATA = 'arx_model_100.ph'
TRAIN_DATA = '../../../Data/MagLev.txt'


def main():

    """
    Main Method.

    :return: none
    """

    # Use trained ARX
    # If using already trained models, then configure
    # models save location above.
    trained_model = False

    # Training Parameters
    epochs, print_every = 1000, 10

    # ARX TDL Sizes for Input and Output
    delays = 6

    # Prediction horizon for training
    prediction_horizon = 100

    # Data path
    path = TRAIN_DATA

    # Import data for model
    inputs, labels, scalar = import_select_data(path)

    # Setup model
    model = ARX(input_delay_size=delays, output_delay_size=delays, input_size=1, output_size=1)

    # Train k-Steps Ahead ARX model
    if not trained_model:

        print()
        print("Training k-Step Ahead ARX for Horizon: " + str(prediction_horizon))
        print("-------------------------------------------------------")

        # Setup Data
        input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                        prediction_horizon=prediction_horizon,
                                                        overlap=10)

        # Train Model
        model, _ = train_scg(model, input_tensor, label_tensor, epochs=epochs,
                             print_every=print_every, plot_every=False)

        # Save Model
        folder_name = ARX_SAVE_LOC + "arx_" + CURRENT_TIME
        file_name = "arx_model_" + str(prediction_horizon) + '.ph'
        save_model(model, folder_name, file_name)

    else:
        model = load_model(model, ARX_LOAD_LOC, ARX_LOAD_DATA)

    # k-step ahead response
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=False)
    input_tensor, label_tensor = input_tensor.view(1, -1, 1), label_tensor.view(1, -1, 1)
    input_tensor, label_tensor = input_tensor[:, :10000, :], label_tensor[:, :10000, :]
    plot_response(model, input_tensor, label_tensor)

    # Plot autocorrelation function for residual errors (prediction errors)
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=True)
    input_tensor, label_tensor = input_tensor[:10000, :, :], label_tensor[:10000, :, :]
    plot_acf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_errors(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_ccf(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])
    plot_error_histogram(model, input_tensor, label_tensor, k_step=[1, 2, 3, 4])

    # Run a skyline function for performance of plant
    plot_skyline_response(model, 10000, 200, 10, 1, -1)

    # Plot the mean square error for the k-step ahead predictions
    input_tensor, label_tensor = init_training_data(model, inputs, labels,
                                                    prediction_horizon=prediction_horizon,
                                                    overlap=False)

    mae = unscaled_prediction_mae(model, input_tensor, label_tensor, scalar)
    plt.plot(np.linspace(1, prediction_horizon, prediction_horizon),
             mae, linewidth=1.5)
    plt.title('Mean Absolute Error for ARX k-Step Ahead Predictor')
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
