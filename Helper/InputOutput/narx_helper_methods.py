from Optimizer.scg import SCG
from Helper.helper_methods import *
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import floor, sqrt, ceil


sns.set()


def init_tdl(model, inputs, labels, device):

    """
    Initialize TDLs using input and label data.

    :param model: NARX model
    :param inputs: inputs
    :param labels: labels
    :param device: device type
    :return: input, labels, input TDL, output TDL
    """

    max_delay_size = max(model.input_delay_size, model.output_delay_size)

    # input tap-delay
    itdl = inputs[:, max_delay_size - model.input_delay_size: max_delay_size, :].to(device)

    # output tap-delay
    otdl = labels[:, max_delay_size - model.output_delay_size: max_delay_size, :].to(device)

    # input and label data
    input = inputs[:, max_delay_size:, :]
    label = labels[:, max_delay_size:, :]

    return input, label, itdl, otdl


def init_tdl_zeros(model, batch_size, device):

    """
    Initialize TDLs with zeros.

    :param model: NARX model
    :param batch_size: size of batch
    :return: input TDL, output TDL
    """

    # input tap-delay
    itdl = torch.zeros((batch_size, model.input_delay_size, model.input_size), dtype=torch.float64).to(device)

    # output tap-delay
    otdl = torch.zeros((batch_size, model.output_delay_size, model.output_size), dtype=torch.float64).to(device)

    return itdl, otdl


def train_scg(model, input_tensor, label_tensor, epochs=1000, print_every=50, plot_every=False,
              number_batches=1, device=None):

    """
    Train NARX model for k-steps ahead using Scaled Conjugate Gradient (SCG).

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param epochs: number of training epochs
    :param print_every: print loss value every N number of iterations
    :param plot_every: plot loss value every N number of iterations
    :param number_batches: number of batches for training. Default: 1.
    :param device: device type
    :return: trained model
    """

    # Initialize Time, Loss Value, and Array of Losses
    start = time.time()
    all_losses = []
    losses = []
    epoch_values = []
    current_loss = 0
    reset_count = 0
    reset = False
    previous_norm_grad = None

    # Performance criterion
    criterion = nn.MSELoss()

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # SCG optimizer
    scg = SCG(model.parameters())

    # Batch iterations
    iter = 200

    # Training Epochs
    for epoch in range(epochs + 1):

        if epoch % iter is 0 or epoch is 0 or reset is True:
            x = torch.randperm(input_tensor.size()[0])
            input_tensor = input_tensor[x]
            label_tensor = label_tensor[x]

            # Setup batch inputs and labels
            n = floor(input_tensor.size()[0] / number_batches)
            begin = 0
            end = n

            batch_inputs = input_tensor[begin: end, :, :]
            batch_labels = label_tensor[begin: end, :, :]

        def closure():

            """
            Closure method calculates loss and gradients after a forward pass
            through network.

            :return: performance (loss)
            """

            scg.zero_grad()
            inputs, labels, itdl, otdl = init_tdl(model, batch_inputs, batch_labels, device)
            outputs, _, _ = model(inputs, itdl, otdl)
            perf = criterion(labels, outputs)
            perf.backward()

            return perf

        # Perform a step of conjugate gradient
        loss = scg.step(closure)
        current_loss += loss

        # Copy optimizer states
        norm_gx = scg.state['norm_gx']

        if previous_norm_grad is not None and abs(previous_norm_grad - norm_gx) <= 1e-12:
            reset_count += 1
        else:
            reset_count = 0

        previous_norm_grad = norm_gx

        if reset_count >= 5:
            reset = True
            reset_count = 0
        else:
            reset = False

        # Average loss value over batches
        current_loss /= float(1)

        # Append loss to array
        losses.append(current_loss)

        # Print results
        if print_every is not False and epoch % print_every is 0 or epochs is epoch:
            print('epoch: %d percentage: %d%%, time: (%s), loss: %.10f, norm gradients: %.10f, count: %s'
                  % (epoch, epoch / epochs * 100.0, time_since(start), current_loss, norm_gx, reset_count))

        # Plot results
        if plot_every is not False and (epoch % plot_every is 0 or epochs is epoch):
            all_losses.append(current_loss)
            epoch_values.append(epoch)

            if epoch is 0:
                plt.figure()

            plt.clf()
            plt.plot(epoch_values, all_losses)
            plt.ylabel('Loss (MSE)')
            plt.xlabel('Epochs')
            plt.yscale('log')
            plt.pause(0.0001)

        # Reset loss value
        current_loss = 0

    return model, losses


def init_training_data(model, inputs, labels, prediction_horizon=1, loading_length=0, overlap=True):

    """
    Initialize Data for Training Model.

    :param model: NARX model
    :param inputs: inputs (numpy array)
    :param labels: labels (numpy array)
    :param prediction_horizon: prediction horizon size
    :param loading_length: time length used for loading the NARX
    :param overlap: sort data with overlap
    :return: sorted inputs and labels as tensors for training
    """

    # Find Max Tap Delay Size
    tap_delay_size = max(model.input_delay_size, model.output_delay_size)

    if overlap is True:

        # Determine Number of Sections for Training
        s = prediction_horizon + tap_delay_size + loading_length
        sub_sections = labels.shape[0] - s
        length = s

        # Determine Length of Sections for Training
        input_sections = np.zeros((sub_sections, length, inputs.shape[1]))
        label_sections = np.zeros((sub_sections, length, labels.shape[1]))

        # Sort Data
        minimum = 0
        maximum = length
        for i in range(sub_sections):
            input_sections[i, :, :] = inputs[minimum: maximum, :]
            label_sections[i, :, :] = labels[minimum: maximum, :]
            minimum += 1
            maximum += 1

    elif overlap is False:

        # Determine Number of Sections for Training
        s = prediction_horizon + tap_delay_size + loading_length
        sub_sections = floor(inputs.shape[0] / s)
        length = s

        # Determine Length of Sections for Training
        input_sections = np.zeros((sub_sections, length, inputs.shape[1]))
        label_sections = np.zeros((sub_sections, length, labels.shape[1]))

        # Sort Data
        for i in range(sub_sections):
            minimum = i * length
            maximum = (i + 1) * length
            input_sections[i, :, :] = inputs[minimum: maximum, :]
            label_sections[i, :, :] = labels[minimum: maximum, :]

    else:

        # Determine Number of Sections for Training
        s = prediction_horizon + tap_delay_size + loading_length
        sub_sections = floor(labels.shape[0] / overlap) - s
        length = s

        # Determine Length of Sections for Training
        input_sections = np.zeros((sub_sections, length, inputs.shape[1]))
        label_sections = np.zeros((sub_sections, length, labels.shape[1]))

        # Sort Data
        minimum = 0
        maximum = length
        for i in range(sub_sections):
            input_sections[i, :, :] = inputs[minimum: maximum, :]
            label_sections[i, :, :] = labels[minimum: maximum, :]
            minimum += overlap
            maximum += overlap

    return torch.tensor(input_sections), torch.tensor(label_sections)


def __multi_step_ahead_prediction(model, input_tensor, label_tensor, loading_length, device):

    """
    Multiple Step Ahead Prediction.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param loading_length: time length used for loading the NARX
    :param device: specified device to use (Default: None - select what is available)
    :return: predictions
    """

    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)

    # Sort data for loading and k-step ahead predictions.
    # The loading is used to pre-fill the error TDLs
    # before preforming a k-step ahead prediction.
    loading_inputs, loading_labels = inputs[:, :loading_length, :], labels[:, :loading_length, :]
    k_step_inputs, k_step_labels = inputs[:, loading_length:, :], labels[:, loading_length:, :]

    # Perform a loading to pre-fill error TDLs
    loading_outputs, itdl, otdl = model.loading(loading_inputs, loading_labels, itdl, otdl)

    # Perform a k-step ahead prediction
    k_step_outputs, _, _ = model(k_step_inputs, itdl, otdl)

    return k_step_outputs, loading_outputs


def prediction_mae(model, input_tensor, label_tensor, loading_length=0, return_loading_error=False, device=None):

    """
    Prediction MAE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading MAE with the multi-step ahead MAE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction mae
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    error = labels - outputs
    error = error.cpu().data.numpy()
    error = error.reshape((error.shape[0], error.shape[1]))
    error = (np.abs(error)).mean(axis=0)

    return error


def prediction_mse(model, input_tensor, label_tensor, loading_length=0, return_loading_error=False, device=None):

    """
    Prediction MSE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading MSE with the multi-step ahead MSE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction mse
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    error = labels - outputs
    error = error.cpu().data.numpy()
    error = error.reshape((error.shape[0], error.shape[1]))
    error = (np.power(error, 2)).mean(axis=0)

    return error


def prediction_rmse(model, input_tensor, label_tensor, loading_length=0, return_loading_error=False, device=None):

    """
    Prediction RMSE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading RMSE with the multi-step ahead RMSE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction rmse
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    error = labels - outputs
    error = error.cpu().data.numpy()
    error = error.reshape((error.shape[0], error.shape[1]))
    error = np.sqrt((np.power(error, 2)).mean(axis=0))

    return error


def unscaled_prediction_mae(model, input_tensor, label_tensor, scalar, loading_length=0, return_loading_error=False,
                            device=None):

    """
    Prediction MAE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param scalar: scalar for transforming output data
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading MAE with the multi-step ahead MAE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction mae
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    labels = labels.cpu().data.numpy()
    labels = labels.reshape((labels.shape[0], labels.shape[1]))
    labels = (labels - scalar.min_[1]) / scalar.scale_[1]

    outputs = outputs.cpu().data.numpy()
    outputs = outputs.reshape((outputs.shape[0], outputs.shape[1]))
    outputs = (outputs - scalar.min_[1]) / scalar.scale_[1]

    error = labels - outputs
    error = (np.abs(error)).mean(axis=0)

    return error


def unscaled_prediction_mse(model, input_tensor, label_tensor, scalar, loading_length=0, return_loading_error=False,
                            device=None):

    """
    Prediction MSE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param scalar: scalar for transforming output data
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading MSE with the multi-step ahead MSE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction mse
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    labels = labels.cpu().data.numpy()
    labels = labels.reshape((labels.shape[0], labels.shape[1]))
    labels = (labels - scalar.min_[1]) / scalar.scale_[1]

    outputs = outputs.cpu().data.numpy()
    outputs = outputs.reshape((outputs.shape[0], outputs.shape[1]))
    outputs = (outputs - scalar.min_[1]) / scalar.scale_[1]

    error = labels - outputs
    error = (np.power(error, 2)).mean(axis=0)

    return error


def unscaled_prediction_rmse(model, input_tensor, label_tensor, scalar, loading_length=0, return_loading_error=False,
                             device=None):

    """
    Prediction RMSE.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :param scalar: scalar for transforming output data
    :param loading_length: time length used for loading the NARX
    :param return_loading_error: return the loading RMSE with the multi-step ahead RMSE
    :param device: specified device to use (Default: None - select what is available)
    :return: prediction rmse
    """

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    if return_loading_error:
        # Combine loading and multi-step predictions/labels
        outputs = torch.cat([loading_outputs, k_step_outputs], dim=1)
        labels = torch.cat([loading_labels, k_step_labels], dim=1)
    else:
        # Use the multi-step predictions/labels
        outputs = k_step_outputs
        labels = k_step_labels

    labels = labels.cpu().data.numpy()
    labels = labels.reshape((labels.shape[0], labels.shape[1]))
    labels = (labels - scalar.min_[1]) / scalar.scale_[1]

    outputs = outputs.cpu().data.numpy()
    outputs = outputs.reshape((outputs.shape[0], outputs.shape[1]))
    outputs = (outputs - scalar.min_[1]) / scalar.scale_[1]

    error = labels - outputs
    error = np.sqrt((np.power(error, 2)).mean(axis=0))

    return error


def plot_response(model, input_tensor, label_tensor, loading_length=0):

    """
    Plot response.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: output tensor
    :param loading_length: time length used for loading the NARX model
    :return: none
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    loading_labels, k_step_labels = labels[:, :loading_length, :], labels[:, loading_length:, :]

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    loading_length, device)

    # Sort data for plotting
    o, l, t = [], [], []
    for i in range(k_step_outputs.size()[1]):
        t.append(i + 1)
        o.append(k_step_outputs[0, i, 0].item())
        l.append(k_step_labels[0, i, 0].item())

    # Plot Data
    plt.clf()

    plt.plot(t, o, color='tab:blue')
    plt.plot(t, l, color='tab:red')
    plt.xlabel('time steps')
    plt.ylabel('outputs', color='tab:blue')
    plt.title('Plant Output Response')

    plt.legend(['Predictions', 'Labels'])

    plt.show()


def plot_errors(model, input_tensor, label_tensor, k_step=[1]):

    """
    Plot errors.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: output tensor
    :param k_step: plot acf at specified k-step prediction (Default 1)
    :return: none
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    k_step_labels = labels

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    0, device)

    # Calculate prediction errors
    errors = k_step_labels - k_step_outputs

    # Plot Data
    plt.clf()
    plt.title('Plant Prediction Errors')

    for z, k in zip(range(len(k_step)), k_step):

        plt.subplot(ceil(len(k_step) / 2), 2, z + 1)

        # Split errors as provided lag
        k_step_minus_one = k - 1
        er = errors[:, k_step_minus_one: k_step_minus_one + 1, :]
        er = er.view(1, -1, model.output_size)

        # Sort data for plotting
        e, t = [], []
        for i in range(er.size()[1]):
            t.append(i + 1)
            e.append(er[0, i, 0].item())

        plt.plot(t, e, color='tab:blue')

        plt.xlabel('Time Steps')
        plt.ylabel('Errors', color='tab:blue')
        plt.title('Errors for ' + str(k) + '-Step Prediction')

    plt.tight_layout()
    plt.show()


def plot_acf(model, input_tensor, label_tensor, k_step=[1], lags=25):

    """
    Plot error autocorrelation function.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: output tensor
    :param k_step: plot acf at specified k-step prediction (Default [1])
    :param lags: significant lags (Default 25)
    :return: none
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    k_step_labels = labels

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    0, device)

    # Calculate prediction errors
    errors = k_step_labels - k_step_outputs

    # Plot Data
    plt.clf()

    for z, k in zip(range(len(k_step)), k_step):

        if len(k_step) > 1:
            plt.subplot(ceil(len(k_step) / 2), 2, z + 1)

        # Split errors as provided lag
        k_step_minus_one = k - 1
        er = errors[:, k_step_minus_one: k_step_minus_one + 1, :]
        er = er.view(1, -1, model.output_size)

        # Sort data for plotting
        e, t = [], []
        for i in range(er.size()[1]):
            t.append(i + 1)
            e.append(er[0, i, 0].item())

        e = np.array(e)
        n = e.size
        if n % 2 is not 0:
            n -= 1
            e = e[:-1]
        variance = e.var()
        x = e - e.mean()

        r = np.correlate(x, x, mode='full')
        acf = r / (variance * n)
        s = np.linspace(-lags, lags, 2 * lags + 1)

        plt.stem(s, acf[n - lags - 1: n + lags], use_line_collection=True)
        plt.axhline(y=1/lags, color='r', linestyle='--')
        plt.axhline(y=-1/lags, color='r', linestyle='--')

        plt.ylabel('ACF', color='tab:blue')
        plt.title('Autocorrelation Function (ACF) for ' + str(k) + '-Step Prediction')

    plt.show()


def plot_error_histogram(model, input_tensor, label_tensor, k_step=[1], number_of_bins=25):

    """
    Plot error histogram.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: output tensor
    :param k_step: plot acf at specified k-step prediction (Default 1)
    :param number_of_bins: number of bins in histogram
    :return: none
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    k_step_labels = labels

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    0, device)

    # Calculate prediction errors
    errors = k_step_labels - k_step_outputs

    # Plot Data
    plt.clf()
    plt.title('Plant Error Histogram')

    for z, k in zip(range(len(k_step)), k_step):

        if len(k_step) > 1:
            plt.subplot(ceil(len(k_step) / 2), 2, z + 1)

        # Split errors as provided lag
        k_step_minus_one = k - 1
        er = errors[:, k_step_minus_one: k_step_minus_one + 1, :]
        er = er.view(1, -1, model.output_size)

        # Sort data for plotting
        e = []
        for i in range(er.size()[1]):
            e.append(er[0, i, 0].item())

        plt.hist(e, color='tab:blue', bins=number_of_bins)

        plt.xlabel('Errors')
        plt.ylabel('Count', color='tab:blue')
        plt.title('Errors for ' + str(k) + '-Step Prediction')

    plt.tight_layout()
    plt.show()


def plot_ccf(model, input_tensor, label_tensor, k_step=[1], lags=25, loading_length=0):

    """
    Plot input and error cross correlation function.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: output tensor
    :param k_step: plot acf at specified k-step prediction (Default [1])
    :param lags: significant lags (Default 25)
    :param loading_length: time length used for loading the NARMAX model's tap-delay of errors. Should usually be
                           3 times the length of the error tap-delay.
    :return: none
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Sort data for loading and k-step ahead predictions.
    inputs, labels, itdl, otdl = init_tdl(model, input_tensor, label_tensor, device)
    k_step_labels = labels
    k_step_inputs = inputs

    # Perform a k-step ahead prediction
    k_step_outputs, loading_outputs = __multi_step_ahead_prediction(model, input_tensor, label_tensor,
                                                                    0, device)

    # Calculate prediction errors
    errors = k_step_labels - k_step_outputs

    # Plot Data
    plt.clf()

    for z, k in zip(range(len(k_step)), k_step):

        if len(k_step) > 1:
            plt.subplot(ceil(len(k_step) / 2), 2, z + 1)

        # Split errors as provided lag
        k_step_minus_one = k - 1
        er = errors[:, k_step_minus_one: k_step_minus_one + 1, :]
        er = er.view(1, -1, model.output_size)
        inp = k_step_inputs[:, k_step_minus_one: k_step_minus_one + 1, :]
        inp = inp.view(1, -1, model.input_size)

        # Sort data for plotting
        e, i, t = [], [], []
        for p in range(er.size()[1]):
            t.append(p + 1)
            e.append(er[0, p, 0].item())
            i.append(inp[0, p, 0].item())

        e = np.array(e)
        i = np.array(i)
        n = e.size
        if n % 2 is not 0:
            n -= 1
            e = e[:-1]
        variance_e = e.var()
        variance_i = i.var()
        x_e = e - e.mean()
        x_i = i - i.mean()

        r = np.correlate(x_e, x_i, mode='full')
        acf = r / (sqrt(variance_e) * sqrt(variance_i) * n)
        s = np.linspace(-lags, lags, 2 * lags + 1)

        plt.stem(s, acf[n - lags - 1: n + lags], use_line_collection=True)
        plt.axhline(y=1 / lags, color='r', linestyle='--')
        plt.axhline(y=-1 / lags, color='r', linestyle='--')

        plt.ylabel('CCF', color='tab:blue')
        plt.title('Cross Correlation Function (CCF) for ' + str(k) + '-Step Prediction')

    plt.show()


def plot_skyline_response(model, points, max_width, min_width, max_height, min_height):

    """
    Plot plant response fro a skyline reference input.

    :param model: model
    :param points: number of points in skyline input
    :param points: number of points in skyline reference input
    :param max_width: max width of the skyline function
    :param min_width: min width of the skyline function
    :param max_height: max amplitude in the skyline function
    :param min_height: min amplitude in the skyline function
    :return: none
    """

    # Create a skyline  input
    input_tensor = skyline(points, max_width, min_width, max_height, min_height)

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor = input_tensor.to(device)

    # Forward pass through network
    itdl, otdl = init_tdl_zeros(model, 1, device)
    itdl, otdl = itdl.to(device), otdl.to(device)
    outputs, _, _ = model(input_tensor, itdl, otdl)

    # Sort data for plotting
    o, t, r = [], [], []
    for i in range(outputs.size()[1]):
        t.append(i + 1)
        o.append(outputs[0, i, 0].item())
        r.append(input_tensor[0, i, 0].item())

    # Plot Data
    plt.clf()

    plt.xlabel('time steps')
    plt.ylabel('outputs and inputs', color='tab:blue')
    plt.title('Plant Output Response')
    plt.plot(t, r, color='tab:red')
    plt.plot(t, o, color='tab:blue')

    plt.legend(['Inputs', 'Predictions'])

    plt.show()
