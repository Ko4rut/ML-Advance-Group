import numpy as np


def temperature_exponential(time, initial_temperature, alpha):
    return initial_temperature * alpha**time


def temperature_linear(time, initial_temperature, eta):
    return initial_temperature - eta * time


def temperature_fast(time, initial_temperature):
    return initial_temperature / (1 + time)


def temperature_logarithmic(time, initial_temperature):
    return initial_temperature / np.log1p(time)