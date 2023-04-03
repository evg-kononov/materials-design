import torch
import numpy as np
from pymoo.core.sampling import Sampling

def criterion_maxmin(X: torch.Tensor):
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    return D.min().item()


def sampling_lhs(n_samples, n_var, xl=0, xu=1, device=None, smooth=True, criterion=criterion_maxmin, n_iter=50):

    X = sampling_lhs_unit(n_samples, n_var, device=device, smooth=smooth)

    # if a criterion is selected to further improve the sampling
    if criterion is not None:

        # current best score is stored here
        score = criterion(X)

        for j in range(1, n_iter):

            # create new random sample and check the score again
            _X = sampling_lhs_unit(n_samples, n_var, device=device, smooth=smooth)
            _score = criterion(_X)

            if _score > score:
                X, score = _X, _score
    X = xl + X * (xu - xl)
    X = X.cpu().numpy()
    return X


def sampling_lhs_unit(n_samples, n_var, device=None, smooth=True):
    X = torch.rand(size=(n_samples, n_var), device=device)
    Xp = torch.argsort(X, dim=0) + 1

    if smooth:
        Xp = Xp - torch.rand(Xp.shape, device=device)
    else:
        Xp = Xp - 0.5
    Xp /= n_samples
    return Xp


class LatinHypercubeSampling(Sampling):

    def __init__(self,
                 device=None,
                 smooth=True,
                 iterations=20,
                 criterion=criterion_maxmin) -> None:
        super().__init__()
        self.smooth = smooth
        self.iterations = iterations
        self.criterion = criterion
        self.device = device

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        xl = torch.from_numpy(xl).to(device=self.device)
        xu = torch.from_numpy(xu).to(device=self.device)

        X = sampling_lhs(n_samples, problem.n_var, xl=xl, xu=xu, device=self.device, smooth=self.smooth,
                         criterion=self.criterion, n_iter=self.iterations)

        return X


class LHS(LatinHypercubeSampling):
    pass
