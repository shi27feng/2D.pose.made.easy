import torch


def l2_loss(in_tenor, target, mask, batch_size):
    loss = .5 * torch.pow(torch.mul(in_tenor - target, mask), 2) / batch_size
    return torch.sum(loss)
