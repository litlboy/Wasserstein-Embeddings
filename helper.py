import numpy as np
import torch

def vec_pairwise_distances(x):
    dist = torch.sum((x[:, None, :, None, :]-x[None, :, None, :, :])**2, dim=4)
    return dist

def vec_sinkhorn_divergence(x, *args):

    # Read params
    L = args[0]
    p = args[1]
    lambd = torch.tensor(args[2], dtype=torch.float64)

    # Initialize dist matrix and K
    D = vec_pairwise_distances(x)**p
    K = torch.exp(-D/lambd)

    # Initialize r and c
    r = torch.ones(D.size(0), D.size(1), D.size(2), dtype=torch.float64)
    c = torch.ones(D.size(0), D.size(1), D.size(3), dtype=torch.float64)

    # Uniform weights
    u = torch.ones(D.size(2), dtype=torch.float64)/D.size(2)
    v = torch.ones(D.size(3), dtype=torch.float64)/D.size(3)

    # Fixed point algo
    for _ in range(L):
        r = u/((K@c[:, :, :, None]).squeeze() + u*(1e-35))
        c = v/((torch.transpose(K, 2, 3)@r[:, :, :, None]).squeeze() + v*(1e-35))

    T = r[:, :, :, None]*K*c[:, :, None, :]

    # Sinkhorn divergence
    return torch.sum(D*T, dim=(2,3))

if __name__=='__main__':

    params = [10, 1, 0.05]
    #x = np.random.randn(100, 2)
    #print(sinkhorn_divergence(torch.tensor(x, dtype=torch.float64),
              #torch.tensor(x, dtype=torch.float64), *params))
    # Ground space dimension
    m = 3

    # Number of dirac masses
    d = 2

    # Number of points to embed
    n = 2

    # Variables
    x = np.random.randn(n, d, m)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float64)

    res = vec_sinkhorn_divergence(x, *params)
    print(res)

