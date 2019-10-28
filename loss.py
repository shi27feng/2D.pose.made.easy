import torch


def l2loss(in_tenor, target, batch_size, mask=None):
    diff = in_tenor - target if mask is None else torch.mul(in_tenor - target, mask)
    loss = .5 * torch.pow(diff, 2) / batch_size
    return torch.sum(loss)
