import paddle
import paddle.nn.functional as F

def tanh_rescale(x, x_min=-1., x_max=1.):
    return (paddle.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)

def arctanh(x, eps=1e-6):
    x = x*(1. - eps)
    return (paddle.log((1 + x) / (1 - x))) * 0.5