import torch
import numpy as np

import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from torch import nn
from network import MappingNetwork, Generator
from training_loop import generate_noise, mixing_regularization, generate_z


def f1(x):
    return torch.sum(x, dim=(1, 2, 3, 4)) / (x.shape[2] * x.shape[3] * x.shape[4])


def f2(x):
    return torch.sum(x, dim=(1, 2, 3, 4)) - torch.sum(x, dim=(1, 2, 3, 4)) ** 2


class OptimalDesign(Problem):
    def __init__(self, net_M, net_G, n_var, n_obj, n_ieq_constr, zl, zu):
        """
        net_M - mapping network
        net_G - generator network
        n_var - number of variables
        n_obj - number of objectives
        n_ieq_constr - number of inequality constraints
        zl - lower bound for the variables
        zu - upper bound for the variables
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=zl, xu=zu)
        self.net_M = net_M
        self.net_G = net_G
        self.n_blocks = self.net_G.n_blocks
        self.init_const_shape = np.array(self.net_G.initial_constant.shape[2:])
        self.device = next(net_M.parameters()).device

    def _evaluate(self, z, out, *args, **kwargs):
        with torch.no_grad():
            z = [torch.from_numpy(z).to(device=self.device, dtype=torch.float32)]
            w = [self.net_M(z_i) for z_i in z]
            w = mixing_regularization(w, self.n_blocks)
            noise = generate_noise(z[0].shape[0], self.n_blocks, self.init_const_shape, self.device)
            x = self.net_G(w, noise)

        x = torch.where(x >= 0, 1., -1.) * 0.5 + 0.5
        f1_value = f1(x).cpu().numpy()
        g1 = -f1_value
        g2 = f1_value - 1
        #g1 = -f1(x)
        #g2 = f1(x) - 1
        #g3 = -f2(x)

        out["F"] = [f1_value]
        out["G"] = [g1, g2]
        #out["F"] = [f1(x), f2(x)]
        #out["G"] = [g1, g2, g3]


if __name__ == "__main__":
    zl, zu = -5, 5
    d_latent = 64
    n_layers = 4
    lr_multiplier = 0.01
    log_resolution = 6
    n_features = 8
    max_features = 64
    activation = nn.Tanh()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_M = MappingNetwork(d_latent, n_layers, lr_multiplier).to(device).eval()
    net_G = Generator(log_resolution, d_latent, n_features, max_features, activation=activation).to(device).eval()

    ckpt_path = r"../checkpoint/008000.pt"
    checkpoint = torch.load(ckpt_path)
    net_M.load_state_dict(checkpoint["net_M_ema"])
    net_G.load_state_dict(checkpoint["net_G_ema"])

    problem = OptimalDesign(net_M=net_M, net_G=net_G, n_var=d_latent, n_obj=1, n_ieq_constr=2, zl=zl, zu=zu)

    sampling = LHS()
    algorithm = NSGA2(pop_size=128, sampling=sampling)

    res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=True, seed=42)
    print("Threads:", res.exec_time)

    plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.title("Objective Space")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()
