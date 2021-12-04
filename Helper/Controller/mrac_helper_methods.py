from Optimizer.scg import SCG
from Helper.helper_methods import *
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import floor


sns.set()


def init_tdl_zeros(batch_size, configuration, device):

    """
    Initialize TDLs with zeros.

    :param batch_size: batch size of inputs and outputs.
    :param configuration: controller configuration
    :param device: device type cpu/gpu
    :return: none
    """

    # CONTROLLER FEEDBACK
    # ---------------------------------------------------------------------------------------
    # Initialize input TDL
    citdl = torch.zeros((batch_size, configuration.controller_input_delay, 1), dtype=torch.float64)
    citdl = citdl.to(device)

    # Initialize reference TDL
    crtdl = torch.zeros((batch_size, configuration.controller_reference_delay, 1), dtype=torch.float64)
    crtdl = crtdl.to(device)

    # Initialize output TDL
    cotdl = torch.zeros((batch_size, configuration.controller_output_delay, 1), dtype=torch.float64)
    cotdl = cotdl.to(device)

    # PLANT FEEDBACK
    # ---------------------------------------------------------------------------------------
    # Initialize input TDL
    pitdl = torch.zeros((batch_size, configuration.plant_input_delay, 1), dtype=torch.float64)
    pitdl = pitdl.to(device)

    # Initialize output TDL
    potdl = torch.zeros((batch_size, configuration.plant_output_delay, 1), dtype=torch.float64)
    potdl = potdl.to(device)

    return citdl, crtdl, cotdl, pitdl, potdl


def train_scg(model, reference_tensor, label_tensor, epochs=1000, print_every=50, plot_every=False,
              number_batches=1, device=None):

    """
    Train mrac controller model using Scaled Conjugate Gradient (SCG).

    :param model: model
    :param reference_tensor: reference tensor
    :param label_tensor: label tensor
    :param epochs: number of training epochs
    :param print_every: print loss value every N number of iterations
    :param plot_every: plot loss value every N number of iterations
    :param number_batches: number of batches for training. Default: 1.
    :param device: tensor device
    :return: trained model
    """

    # Initialize Time, Loss Value, and Array of Losses
    start = time.time()
    all_losses = []
    losses = []
    epoch_values = []
    current_loss = 0
    reset = False
    reset_count = 0
    previous_norm_grad = None

    # Performance criterion
    criterion = nn.MSELoss()

    # Create Network on GPU/CPU
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    reference_tensor, label_tensor = reference_tensor.to(device), label_tensor.to(device)

    # SCG optimizer
    scg = SCG(model.parameters())

    # Batch iterations
    iter = 2000

    # Training Epochs
    for epoch in range(epochs + 1):

        if epoch % iter is 0 or epoch is 0 or reset is True:
            x = torch.randperm(reference_tensor.size()[0])
            reference_tensor = reference_tensor[x]
            label_tensor = label_tensor[x]

            # Setup batch inputs and labels
            n = floor(reference_tensor.size()[0] / number_batches)
            begin = 0
            end = n

            batch_references = reference_tensor[begin: end, :, :]
            batch_labels = label_tensor[begin: end, :, :]

        def closure():

            """
            Closure method calculates loss and gradients after a forward pass
            through network.

            :return: performance (loss)
            """

            scg.zero_grad()

            # Perform a k-step ahead prediction
            citdl, crtdl, cotdl, pitdl, potdl = init_tdl_zeros(batch_references.size()[0], model.configuration, device)
            output, _, _, _, _, _ = model(batch_references, citdl, crtdl, cotdl, pitdl, potdl)

            perf = criterion(batch_labels, output)
            perf.backward()

            return perf

        # Perform a step of conjugate gradient
        loss = scg.step(closure)
        current_loss += loss

        # Copy optimizer states
        norm_gx = scg.state['norm_gx'].cpu().data.numpy()

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


def init_training_data(inputs, labels, prediction_horizon=1, overlap=True):

    """
    Initialize Data for Training Model.

    :param inputs: inputs (numpy array)
    :param labels: labels (numpy array)
    :param prediction_horizon: prediction horizon size
    :param overlap: sort data with overlap
    :return: sorted inputs and labels as tensors for training
    """

    # Find Max Tap Delay Size
    delay_size = 0

    if overlap is True:

        # Determine Number of Sections for Training
        s = prediction_horizon + delay_size
        sub_sections = labels.shape[0] - s
        length = s

        # Determine Length of Sections for Training With Overlap
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
        s = prediction_horizon + delay_size
        sub_sections = math.floor(inputs.shape[0] / s)
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
        s = prediction_horizon + delay_size
        sub_sections = floor((labels.shape[0] - s) / overlap)
        length = s

        # Determine Length of Sections for Training With Overlap
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


def prediction_error(model, input_tensor, label_tensor):

    """
    Prediction Error.

    :param model: model
    :param input_tensor: input tensor
    :param label_tensor: label tensor
    :return: prediction errors
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    input_tensor, label_tensor = input_tensor.to(device), label_tensor.to(device)

    # Perform a forward pass
    citdl, crtdl, cotdl, pitdl, potdl = init_tdl_zeros(input_tensor.size()[0], model.configuration, device)
    output_tensor, _, _, _, _, _ = model(input_tensor, citdl, crtdl, cotdl, pitdl, potdl)
    error = label_tensor - output_tensor

    error = error.cpu().data.numpy()
    error = error.reshape((error.shape[0], error.shape[1]))
    error = (error ** 2).mean(axis=0)

    return error


def plot_response(model, references, labels):

    """
    Plot controller response.

    :param model: model
    :param references: reference tensor
    :param labels: label tensor
    :return:
    """

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    references_tensor = references.to(device)
    label_tensor = labels.to(device)

    # Perform a forward pass
    citdl, crtdl, cotdl, pitdl, potdl = init_tdl_zeros(references_tensor.size()[0], model.configuration, device)
    output_tensor, _, _, _, _, _ = model(references_tensor, citdl, crtdl, cotdl, pitdl, potdl)

    # Sort data for plotting
    o, l, t, d = [], [], [], []
    for i in range(output_tensor.size()[1]):
        t.append(i + 1)
        o.append(output_tensor[0, i, 0].item())
        l.append(references_tensor[0, i, 0].item())
        d.append(label_tensor[0, i, 0].item())

    # Plot Data
    plt.clf()

    plt.xlabel('time steps')
    plt.ylabel('outputs', color='tab:blue')
    plt.title('Controller Output Response')
    plt.plot(t, l, color='tab:red')
    plt.plot(t, d, color='tab:orange')
    plt.plot(t, o, color='tab:blue')

    plt.legend(['Reference', 'Desired', 'Response'])

    plt.show()


def plot_skyline_response(model, points, max_width, min_width, max_height, min_height, time_constant, h):

    """
    Plot controller response fro a skyline reference input.

    :param model: model
    :param points: number of points in skyline reference input
    :param max_width: max width of the skyline function
    :param min_width: min width of the skyline function
    :param max_height: max amplitude in the skyline function
    :param min_height: min amplitude in the skyline function
    :param time_constant: time constant for first order response
    :param h: sampling rate for first order response
    :return: none
    """

    # Create a skyline reference input
    reference = skyline(points, max_width, min_width, max_height, min_height)
    label = first_order_response(reference, time_constant=time_constant, h=h)

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    reference = reference.to(device)

    # Perform a forward pass
    citdl, crtdl, cotdl, pitdl, potdl = init_tdl_zeros(reference.size()[0], model.configuration, device)
    output_tensor, _, _, _, _, _ = model(reference, citdl, crtdl, cotdl, pitdl, potdl)

    # Sort data for plotting
    o, l, t, d = [], [], [], []
    for i in range(output_tensor.size()[1]):
        t.append(i + 1)
        o.append(output_tensor[0, i, 0].item())
        l.append(reference[0, i, 0].item())
        d.append(label[0, i, 0].item())

    # Plot Data
    plt.clf()

    plt.xlabel('time steps')
    plt.ylabel('outputs', color='tab:blue')
    plt.title('Controller Output Response')
    plt.plot(t, l, color='tab:red')
    plt.plot(t, d, color='tab:orange')
    plt.plot(t, o, color='tab:blue')

    plt.legend(['Reference', 'Desired', 'Response'])

    plt.show()


def plot_simple_skyline_response(model, points, max_height, min_height, time_constant, h):

    """
    Plot controller response fro a skyline reference input.

    :param model: model
    :param points: number of points in skyline reference input
    :param max_height: max amplitude in the skyline function
    :param min_height: min amplitude in the skyline function
    :param time_constant: time constant for first order response
    :param h: sampling rate for first order response
    :return: none
    """

    # Create a skyline reference input
    reference = simple_skyline(points, max_height, min_height)
    label = first_order_response(reference, time_constant=time_constant, h=h)

    # Create Network on GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Data on GPU/CPU
    reference = reference.to(device)

    # Calculate the saliency maps
    citdl, crtdl, cotdl, pitdl, potdl = init_tdl_zeros(reference.size()[0], model.configuration, device)
    output_tensor, _, _, _, _, _ = model(reference, citdl, crtdl, cotdl, pitdl, potdl)

    # Sort data for plotting
    o, l, t, d = [], [], [], []
    for i in range(output_tensor.size()[1]):
        t.append(i + 1)
        o.append(output_tensor[0, i, 0].item())
        l.append(reference[0, i, 0].item())
        d.append(label[0, i, 0].item())

    # Plot Data
    plt.clf()

    plt.xlabel('time steps')
    plt.ylabel('outputs', color='tab:blue')
    plt.title('Controller Output Response')
    plt.plot(t, l, color='tab:red')
    plt.plot(t, d, color='tab:orange')
    plt.plot(t, o, color='tab:blue')

    plt.legend(['Reference', 'Desired', 'Response'])

    plt.show()


def simple_skyline(points, max_height, min_height):

    """
    Generates a simple skyline output tensor. The user can provide
    the max & min of the amplitudes.

    :param points: number of points in the skyline function
    :param max_height: max amplitude in the skyline function
    :param min_height: min amplitude in the skyline function
    :return: skyline output tensor
    """

    w, h = [], []
    w.append(points)
    h.append(0)
    w.append(points)
    h.append(max_height)
    w.append(points)
    h.append(0)
    w.append(points)
    h.append(min_height)
    w.append(points)
    h.append(0)

    start = 0
    signal = torch.zeros((1, 5 * points, 1), dtype=torch.float64)
    for i, j in zip(w, h):
        line = j * torch.ones((1, i, 1), dtype=torch.float64)
        signal[:, start: start + i, :] = line[:, :, :]
        start += i

    return signal
