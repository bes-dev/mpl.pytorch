import torch
import _mpl


class MaxPoolingLoss(object):
    """Max-Pooling Loss

    Implementation of "Loss Max-Pooling for Semantic Image Segmentation"
    https://arxiv.org/abs/1704.02966
    """
    def __init__(self, ratio=0.3, p=1.7, reduce=True):
        """Create a Max-Pooling Loss function

        Parameters
        ----------
        ratio : float
            Minimum percentage of pixels that should be supported by the optimal
            weighting function. Should be in range [0, 1].
        p : float
            p-norm of a uniform weighting function.
        reduce : bool
            Reduce loss after re-weighting.
        """
        assert ratio > 0 and ratio <= 1, "ratio should be in range [0, 1]"
        assert p > 1, "p should be >1"
        self.ratio = ratio
        self.p = p
        self.reduce = reduce


    def __call__(self, loss):
        is_cuda = loss.is_cuda
        shape = loss.size()
        loss = loss.view(-1)
        losses, indices = loss.sort()
        losses = losses.cpu()
        indices = indices.cpu()
        weights = torch.zeros(losses.size(0))
        _mpl.compute_weights(losses.size(0), losses, indices, weights, self.ratio, self.p)
        loss = loss.view(shape)
        weights = weights.view(shape)
        if is_cuda:
            weights = weights.cuda()
        loss = weights * loss
        if self.reduce:
            loss = loss.sum()
        return loss
