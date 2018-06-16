import torch

def compute_weights(losses, indices, weights, ratio, p):
    size = losses.size(0)

    # find first nonzero element
    pos = 0
    while losses[pos]< 1e-5:
        pos += 1
    n = size - pos
    m = int(ratio * n)
    if n <= 0 or m <= 0:
        raise ValueError
    q = p / (p - 1.0)
    c = m - n + 1
    a = [0.0 , 0.0]
    i = pos
    nu = 0.0
    while i < n and nu < 1e-5:
        loss_q = (losses[i] / losses[size - 1]) ** q
        a[0] = a[1]
        a[1] += loss_q
        c += 1
        nu = c * loss_q - a[1]

    # compute alpha
    if nu < 1e-5:
        i += 1
        c += 1
        a[0] = a[1]
    alpha = (a[0] / c) ** (1 / q) * losses[size - 1]

    # compute_weights
    tau = 1.0 / (n ** (1.0 / q)*(m **(1.0 / p)))
    k = i
    while k < n:
        # maybe wrong
        weights[indices[k]] = tau
        k += 1
    if alpha > -1e-5:
        k = pos
        while k < i:
            weights[indices[k]] = tau * (losses[k] / alpha) ** (q - 1)
            k += 1
