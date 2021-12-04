import torch
import os
import numpy as np
from math import floor
import time


def skyline(points, max_width, min_width, max_height, min_height):

    """
    Generates a skyline output tensor. The user can provide
    the max & min of the amplitudes and max & min length of
    the amplitudes.

    :param points: number of points in the skyline function
    :param max_width: max width of the skyline function
    :param min_width: min width of the skyline function
    :param max_height: max amplitude in the skyline function
    :param min_height: min amplitude in the skyline function
    :return: skyline output tensor
    """

    width = max_width - min_width
    height = max_height - min_height

    w, h = [], []
    count = 0
    total = 0
    while total < points:
        w.append(floor(np.random.rand(1)[0] * width + min_width))
        h.append(np.random.rand(1)[0] * height + min_height)

        total += w[count]
        count += 1

    w[count-1] -= (total - points)

    start = 0
    signal = torch.zeros((1, points, 1), dtype=torch.float64)
    for i, j in zip(w, h):
        line = j * torch.ones((1, i, 1), dtype=torch.float64)
        signal[:, start: start + i, :] = line[:, :, :]
        start += i

    return signal


def first_order_response(input_tensor, time_constant, dc_gain=1, h=1e-6):

    """
    First order response.

    :param input_tensor: input tensor
    :param time_constant: time constant in first order system
    :param dc_gain: dc gain between input and output
    :param h: difference coefficient
    :return: output tensor
    """

    a = h / (time_constant + h)

    n = input_tensor.size()[1]
    output_tensor = torch.zeros(1, n, 1)
    output_tensor[:, 0, :] = a * input_tensor[:, 0, :]
    for i in range(1, n):
        output_tensor[:, i, :] = (1. - a) * output_tensor[:, i - 1, :] + dc_gain * a * input_tensor[:, i, :]

    return output_tensor


def time_since(since):

    """
    Time since last value

    :param since: last time
    :return: current time
    """

    now = time.time()
    s = now - since
    m = floor(s / 60.)
    s -= m * 60.
    return '%dm %ds' % (m, s)


def save_model(model, directory, file):

    """
    Save model.

    :param model: model
    :param directory: directory for save file
    :param file: save file
    :return: none
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(directory + '/' + file):
        open(directory + '/' + file, 'wb')

    torch.save(model.state_dict(), directory + '/' + file)


def load_model(model, directory, file):

    """
    Load model.

    :param model: model
    :param directory: directory for save file
    :param file: save file
    :return: loaded model
    """

    filename = directory + '/' + file

    if not os.path.exists(filename):
        raise TypeError("Saved directory '%s' doesn't exist." % filename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.load_state_dict(torch.load(filename, map_location=device))

    return model
