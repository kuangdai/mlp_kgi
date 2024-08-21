import torch


def uniform(sizes, low, high):
    """
    Sample from uniform distribution U(low, high)
    :param sizes: sizes of output
    :param low: lower bound
    :param high: upper bound
    :return: sampled tensor
    """
    return torch.rand(sizes) * (high - low) + low


def kgi_layer(layer, knot_low=0.1, knot_high=0.9,
              perturb_factor=0.2, kgi_by_bias=True):
    """
    Apply KGI to a layer
    :param layer: the target `nn.Linear`
    :param knot_low: lower bound of knot positions
    :param knot_high: upper bound of knot positions
    :param perturb_factor: factor to perturb the KGI-based weight or bias
    :param kgi_by_bias: whether to achieve KGI by bias (`True`) or weight (`False`)
    :return: the re-initialized `nn.Linear`
    """
    # get original
    n, m = layer.weight.data.shape
    w0 = layer.weight.data
    b0 = layer.bias.data

    # for numerical stability, we do not allow zero bias
    if torch.all(torch.abs(b0) < torch.finfo(b0.dtype).eps):
        bound = torch.sqrt(3. / torch.tensor(m))  # use He
        b0 = uniform(n, -bound, bound)
        layer.bias.data = b0

    # sample knots
    x_knot = uniform(m, low=knot_low, high=knot_high)

    if kgi_by_bias:
        # compute b by KGI
        b_kgi = -torch.mv(w0, x_knot)
        # perturb b
        layer.bias.data = (1 - perturb_factor) * b_kgi + perturb_factor * b0
    else:
        # compute w by KGI
        x2 = torch.dot(x_knot, x_knot)
        wp = -torch.outer(b0, x_knot) / x2  # particular solution
        wh = w0 - torch.outer(torch.mv(w0, x_knot), x_knot) / x2  # homogenous solution
        if m > 1:
            # combine the two by L1
            alpha = (w0.abs().sum() - wp.abs().sum()) / wh.abs().sum()
            w_kgi = wp + alpha * wh
        else:
            w_kgi = wp  # in this case, wh = 0
        # perturb w
        layer.weight.data = (1 - perturb_factor) * w_kgi + perturb_factor * w0
    return layer


def kgi_model(model, knot_low=0.1, knot_high=0.9,
              perturb_factor=0.2, kgi_by_bias=True):
    """
    Apply KGI to a model
    :param model: the target `nn.Module`
    :param knot_low: lower bound of knot positions
    :param knot_high: upper bound of knot positions
    :param perturb_factor: factor to perturb the KGI-based weight or bias
    :param kgi_by_bias: whether to achieve KGI by bias (`True`) or weight (`False`)
    :return: the re-initialized `nn.Module`
    """
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            kgi_layer(layer, knot_low, knot_high,
                      perturb_factor, kgi_by_bias)
