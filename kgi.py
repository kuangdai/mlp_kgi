from typing import Any

import torch


def uniform(sizes: list[Any], low: float = 0., high: float = 1.):
    """
    Sample from uniform distribution U(low, high)
    :param sizes: sizes of output
    :param low: lower bound
    :param high: upper bound
    :return: sampled tensor
    """
    return torch.rand(sizes) * (high - low) + low


def kgi_with_bias(fc: torch.nn.Linear,
                  knot_low: float = 0.1, knot_high: float = 0.9,
                  bias_perturb_factor: float = 0.2,
                  input_size_scaling: float = 3.):
    """
    :param fc: the target `nn.Linear` layer
    :param knot_low: lower bound of knot positions
    :param knot_high: upper bound of knot positions
    :param bias_perturb_factor: factor to perturb computed biases
    :param input_size_scaling: factor to divide input size when sampling weights
    :return: the re-initialized `nn.Linear` layer
    """
    n, m = fc.weight.data.shape
    # sample weights
    w_scale = 1 / torch.sqrt(torch.tensor(m) / input_size_scaling)
    w = uniform([n, m], low=-w_scale, high=w_scale)
    # sample knots
    x_knot = uniform(m, low=knot_low, high=knot_high)
    # compute b
    b_exact = -torch.mv(w, x_knot)
    # perturb b
    b_scale = bias_perturb_factor * w_scale
    b_perturbed = uniform(n, low=-b_scale, high=b_scale)
    b = b_exact + b_perturbed
    # assign to layer
    fc.weight.data = w
    fc.bias.data = b
    return fc
