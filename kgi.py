import warnings

import torch

warnings.simplefilter("once", UserWarning)


def apply_kgi_to_layer(layer, knot_low=None, knot_high=None,
                       sampled_inputs=None, clip_ratio=0.1,
                       perturb_factor=0.2, kgi_by_bias=False):
    """
    Apply KGI to a layer
    :param layer: the target `nn.Linear`
    :param knot_low: lower bound of knot positions
    :param knot_high: upper bound of knot positions
    :param sampled_inputs: sampled inputs for automatically determining bounds
    :param clip_ratio: ratio to clip sampled inputs at both min/max ends
    :param perturb_factor: factor to perturb the KGI-based weight or bias
    :param kgi_by_bias: whether to achieve KGI by bias (`True`) or weight (`False`)
    :return: the re-initialized `nn.Linear`
    """
    # get original
    n, m = layer.weight.data.shape
    w0 = layer.weight.data
    b0 = layer.bias.data if layer.bias is not None else torch.zeros(n)

    # sample knots
    if sampled_inputs is not None:
        # automatically determine bounds
        assert knot_low is None and knot_high is None, \
            "When `sampled_inputs` is provided, `knot_low` and `knot_high` must be `None`."
        min_ = sampled_inputs.min()
        max_ = sampled_inputs.max()
        margin = (max_ - min_) * clip_ratio
        knot_low = min_ + margin
        knot_high = max_ - margin
    else:
        assert knot_low is not None and knot_high is not None, \
            "When `sampled_inputs` is `None`, `knot_low` and `knot_high` must be provided."
    x_knot = torch.rand(m) * (knot_high - knot_low) + knot_low

    if kgi_by_bias:
        # compute b by KGI
        b_kgi = -torch.mv(w0, x_knot)
        # perturb b
        if layer.bias is None:
            layer.bias = torch.nn.Parameter(b0)
            warnings.warn("KGI has added bias to a layer created without bias.", UserWarning)
        layer.bias.data = (1 - perturb_factor) * b_kgi + perturb_factor * b0
    else:
        # compute w by KGI
        x2 = torch.dot(x_knot, x_knot)
        wp = -torch.outer(b0, x_knot) / x2  # particular solution
        wh = w0 - torch.outer(torch.mv(w0, x_knot), x_knot) / x2  # homogenous solution
        w_kgi = wp + wh
        # perturb w
        layer.weight.data = (1 - perturb_factor) * w_kgi + perturb_factor * w0


def apply_kgi_to_model(model, knot_low=None, knot_high=None,
                       sampled_inputs=None, clip_ratio=0.1,
                       perturb_factor=0.2, kgi_by_bias=False):
    """
    Apply KGI to a model
    :param model: the target `nn.Module`
    :param knot_low: lower bound of knot positions
    :param knot_high: upper bound of knot positions
    :param sampled_inputs: sampled inputs for automatically determining bounds
    :param clip_ratio: ratio to clip sampled inputs at both min/max ends
    :param perturb_factor: factor to perturb the KGI-based weight or bias
    :param kgi_by_bias: whether to achieve KGI by bias (`True`) or weight (`False`)
    :return: the re-initialized `nn.Module`
    """
    if sampled_inputs is None:
        assert knot_low is not None and knot_high is not None, \
            "When `sampled_inputs` is `None`, `knot_low` and `knot_high` must be provided."
        # handle list of knot_low and knot_high
        n_linear = 0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                n_linear += 1
        if isinstance(knot_low, list):
            assert len(knot_low) == n_linear, "Invalid number of elements in `knot_low`"
        else:
            knot_low = [knot_low] * n_linear
        if isinstance(knot_high, list):
            assert len(knot_high) == n_linear, "Invalid number of elements in `knot_high`"
        else:
            knot_high = [knot_high] * n_linear

        # KGI
        loc = 0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                apply_kgi_to_layer(layer, knot_low=knot_low[loc], knot_high=knot_high[loc],
                                   perturb_factor=perturb_factor, kgi_by_bias=kgi_by_bias)
                loc += 1
        return

    ###################################
    # automatic bounds layer by layer #
    ###################################
    assert knot_low is None and knot_high is None, \
        "When `sampled_inputs` is provided, `knot_low` and `knot_high` must be `None`."
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Linear):
            apply_kgi_to_layer(layer, sampled_inputs=sampled_inputs, clip_ratio=clip_ratio,
                               perturb_factor=perturb_factor, kgi_by_bias=kgi_by_bias)
        sampled_inputs = layer.forward(sampled_inputs)
