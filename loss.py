import torch


def l2loss(in_tenor, target, batch_size):
    diff = in_tenor - target
    loss = .5 * torch.pow(diff, 2) / batch_size
    return torch.sum(loss)


def masked_l2loss(in_tenor, target, batch_size, mask):
    diff = torch.mul(in_tenor - target, mask)
    loss = .5 * torch.pow(diff, 2) / batch_size
    return torch.sum(loss)
