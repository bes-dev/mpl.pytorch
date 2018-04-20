import torch
import _mpl

def MaxPoolingLoss(loss, ratio=0.33, p=1.7, reduce=True):
    shape = loss.size()
    loss = loss.view(-1)
    losses, indices = loss.sort()
    loss = loss.cpu()
    losses = losses.cpu()
    indices = indices.cpu()
    weights = torch.zeros(losses.size(0))
    _mpl.compute_weights(losses, indices, weights, ratio, p)
    loss = loss.view(shape)
    weights = weights.view(shape)
    loss = weights * loss
    if reduce:
        loss = loss.sum()
    return loss
