import re
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lhs import LHS
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from torch import nn
from network import MappingNetwork, Generator, StyleBlock
from training_loop import generate_noise, mixing_regularization, generate_z, data_plot

#from util import expressions
from util.wolfram import inp_preparation



def print_noise_scales(net_G):
    for m in net_G.modules():
        if isinstance(m, StyleBlock):
            print(m.scale_noise)


def generate(net_M, net_G, var, var_shapes, opt_space, block_zero_noise=None):
    n_blocks = net_G.n_blocks
    var = np.expand_dims(var, 0)
    device = next(net_G.parameters()).device

    with torch.no_grad():
        if opt_space == "w+noise":
            zero_noise = generate_noise(1, n_blocks, blocks_zero_noise=[i for i in range(n_blocks)])
            zero_noise = [zero_noise[i].to(device) for i in blocks_zero_noise]
            var = torch.from_numpy(var).to(device=device, dtype=torch.float32)
            w_noise = decompose_var(var, var_shapes)
            w, noise = w_noise[0], w_noise[1:] + zero_noise
            w = mixing_regularization([w], n_blocks)
        if opt_space == "w":
            noise = generate_noise(1, n_blocks, device=device, blocks_zero_noise=[i for i in range(n_blocks)])
            w = [torch.from_numpy(var).to(device=device, dtype=torch.float32)]
            w = mixing_regularization(w, n_blocks)
        if opt_space == "z":
            noise = generate_noise(1, n_blocks, device=device, blocks_zero_noise=[i for i in range(n_blocks)])
            z = [torch.from_numpy(var).to(device=device, dtype=torch.float32)]
            w = [net_M(z_i) for z_i in z]
            w = mixing_regularization(w, n_blocks)

        x = net_G(w, noise)
        x = torch.where(x >= 0, 1., -1.) * 0.5 + 0.5
        x = x.cpu().numpy()
        data_plot(x[0, 0])
        plt.show()


def get_boundaries(opt_space, lower=-3, upper=3, var_shapes=None, net_M=None):
    """
    opt_space - optimization space, either "z" or "z+noise" or "w or "w+noise"
    lower - base lower bound for the variables (usually min of N(0, 1))
    upper - base upper bound for the variables (usually max of N(0, 1))
    var_shapes - variable lengths in the optimizing space
    net_M - mapping network to determine the boundaries of the "w" or "w+noise" space
    """

    def w_boundaries(size=int(10e5)):
        """
        size - amount of generated noise (the more, the more accurately the [min, max] range is determined)
        """
        with torch.no_grad():
            w = net_M((upper - lower) * torch.rand(size, var_shapes[0]) + lower)
            wl, wu = w.min().item(), w.max().item()
            return wl, wu

    if opt_space == "z" or opt_space == "z+noise":
        zl, zu = lower, upper
        return zl, zu

    elif opt_space == "w":
        wl, wu = w_boundaries()
        return wl, wu

    elif opt_space == "w+noise":
        wl, wu = w_boundaries()
        wnl = var_shapes[0] * [wl] + np.sum(var_shapes[1:]) * [lower]
        wnu = var_shapes[0] * [wu] + np.sum(var_shapes[1:]) * [upper]
        return wnl, wnu


def decompose_var(var, shapes):
    """
    var - flat array, which should be decomposed into arrays of different shapes
    shapes - target shapes for decompose
    """
    batch_size = var.shape[0]
    lengths = [np.prod(shape) // batch_size for shape in shapes]  # not including batch_size
    decomposed_var = []
    shift = 0
    for shape, length in zip(shapes, lengths):
        decomposed_var.append(var[:, shift:length + shift].reshape(shape))
        shift += length
    return decomposed_var


def f1(x):
    result = np.sum(x, axis=(1, 2, 3)) / np.prod(x.shape[1:])
    return result


def f2(x):
    # Convert a voxel object into a finite element model
    inp_paths = inp_preparation(x, poolsize=12)
    load_path = "inp_paths.txt"
    with open(load_path, "w+") as f:
        for inp_path in inp_paths:
            f.write(inp_path + "\n")

    if os.path.exists("young_modules.csv"):
        os.remove("young_modules.csv")
    if os.path.exists("sys_exit.txt"):
        os.remove("sys_exit.txt")
    start_idx = 0
    # Define the file path to save the young modules
    save_path = "young_modules.csv"
    abaqus = "abaqus"
    if "ABAQUS_BAT_PATH" in os.environ.keys():
        abaqus = os.environ["ABAQUS_BAT_PATH"]
    abaqus_script_path = "util/abaqus_script.py"
    while True:
        if os.path.exists("sys_exit.txt"):
            with open("sys_exit.txt", "r+") as f:
                start_idx = f.readline()
                print(start_idx)
            if start_idx == "end":
                break
            else:
                start_idx = int(start_idx) + 1
        # Run the Abaqus
        args = " ".join([str(start_idx), load_path, save_path])
        os.system(f"{abaqus} cae noGUI={abaqus_script_path} -- {args}")

    # Load the saved young modules and return them
    young_modules = pd.read_csv(save_path, header=None, names=["inp", "young_module", "fem_volume_fraction"])
    young_modules["inp"] = young_modules["inp"].apply(
        lambda x: int(re.findall(r"\d+", re.findall(r"structure_\d+", x)[0])[0])
    )
    young_modules_dict = {key: value for key, value in zip(young_modules["inp"], young_modules["young_module"])}
    fem_vf_dict = {key: value for key, value in zip(young_modules["inp"], young_modules["fem_volume_fraction"])}

    young_modules_result = np.array([young_modules_dict.get(i, 0.) for i in range(len(x))])
    fem_vf_result = np.array([fem_vf_dict.get(i, 0.) for i in range(len(x))])
    return young_modules_result, fem_vf_result


class OptimalDesign(Problem):
    def __init__(self, batch_size, net_M, net_G, n_var, n_obj, n_ieq_constr, zl, zu, opt_space, blocks_zero_noise):
        """
        batch_size - the number of parallel calculations in optimization space
        net_M - mapping network
        net_G - generator network
        n_var - number of variables
        n_obj - number of objectives
        n_ieq_constr - number of inequality constraints
        zl - lower bound for the variables
        zu - upper bound for the variables
        opt_space - optimization space, either "z" or "z+noise" or "w or "w+noise"
        blocks_zero_noise - generator blocks where noise is zero
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=zl, xu=zu)
        self.batch_size = batch_size
        self.net_M = net_M
        self.net_G = net_G
        self.n_blocks = net_G.n_blocks
        self.device = next(net_M.parameters()).device
        self.opt_space = opt_space
        if opt_space in ["z", "w"]:
            self.zero_noise = generate_noise(
                batch_size, self.n_blocks, device=self.device, blocks_zero_noise=[i for i in range(self.n_blocks)]
            )
        if opt_space in ["z+noise", "w+noise"]:
            d_latent = list(next(self.net_M.parameters()).shape)[0]
            noise_shapes = [i.shape for i in generate_noise(batch_size, self.n_blocks)]
            noise_shapes = [noise_shape for i, noise_shape in enumerate(noise_shapes) if i not in blocks_zero_noise]
            self.var_shapes = [[self.batch_size, d_latent]] + noise_shapes
            self.zero_noise = generate_noise(
                batch_size, self.n_blocks, blocks_zero_noise=[i for i in range(self.n_blocks)]
            )
            self.zero_noise = [self.zero_noise[i].to(self.device) for i in blocks_zero_noise]

    def z_evaluate(self, z):
        with torch.no_grad():
            z = [torch.from_numpy(z).to(device=self.device, dtype=torch.float32)]
            w = [self.net_M(z_i) for z_i in z]
            w = mixing_regularization(w, self.n_blocks)
            x = self.net_G(w, self.zero_noise)
        return x

    def z_noise_evaluate(self, z_noise):
        with torch.no_grad():
            z_noise = torch.from_numpy(z_noise).to(device=self.device, dtype=torch.float32)
            z_noise = decompose_var(z_noise, self.var_shapes)
            z, noise = z_noise[0], z_noise[1:] + self.zero_noise
            w = [self.net_M(z)]
            w = mixing_regularization(w, self.n_blocks)
            x = self.net_G(w, noise)
        return x

    def w_evaluate(self, w):
        with torch.no_grad():
            w = [torch.from_numpy(w).to(device=self.device, dtype=torch.float32)]
            w = mixing_regularization(w, self.n_blocks)
            x = self.net_G(w, self.zero_noise)
        return x

    def w_noise_evaluate(self, w_noise):
        with torch.no_grad():
            w_noise = torch.from_numpy(w_noise).to(device=self.device, dtype=torch.float32)
            w_noise = decompose_var(w_noise, self.var_shapes)
            w, noise = w_noise[0], w_noise[1:] + self.zero_noise
            w = mixing_regularization([w], self.n_blocks)
            x = self.net_G(w, noise)
        return x

    def _evaluate(self, z, out, *args, **kwargs):
        if self.opt_space == "z":
            x = self.z_evaluate(z)
        elif self.opt_space == "z+noise":
            x = self.z_noise_evaluate(z)
        elif self.opt_space == "w":
            x = self.w_evaluate(z)
        elif self.opt_space == "w+noise":
            x = self.w_noise_evaluate(z)
        x = torch.where(x >= 0, 1., -1.) * 0.5 + 0.5
        x = x.squeeze(1).cpu().numpy()

        f1_value = f1(x)
        f2_value, f2_fem_vf = f2(x)

        # out["F"] = [f1_value]
        # out["G"] = [g1, g2]
        out["F"] = [f1_value, -f2_value]
        out["G"] = [-f1_value, f1_value - 1, -f2_value]
        """
        Можно просто добавить 4-ое ограничение:
        g4 = np.abs(1 - f1_value / f2_fem_vf) * 100 - 6
        То есть разница между объемной долей в FEM и VOX не различалась больше, чем на 6%
        Ещё можно попробовать вместо f1_value использовать f2_fem_vf
        """


def main():
    batch_size = 100
    d_latent = 64
    n_layers = 4
    lr_multiplier = 0.01
    log_resolution = 6
    n_features = 8
    max_features = 64
    activation = nn.Tanh()

    net_M = MappingNetwork(d_latent, n_layers, lr_multiplier).eval()
    net_G = Generator(log_resolution, d_latent, n_features, max_features, activation=activation).eval()

    ckpt_path = r"../checkpoint/009500.pt"
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    net_M.load_state_dict(checkpoint["net_M_ema"])
    net_G.load_state_dict(checkpoint["net_G_ema"])

    opt_space = "w+noise"
    if opt_space == "z" or opt_space == "w":
        n_var = d_latent
        lower_bound, upper_bound = get_boundaries(opt_space, net_M=net_M)
    elif opt_space == "z+noise" or opt_space == "w+noise":
        noise = generate_noise(1, net_G.n_blocks, zero=True)
        noise_shapes = [np.prod(list(i.shape)) for i in noise]
        # noise_shapes[0] //= 2  # the first generator block needs only one noise
        var_shapes = [d_latent] + noise_shapes
        n_var = np.sum(var_shapes)
        lower_bound, upper_bound = get_boundaries(opt_space, var_shapes=var_shapes, net_M=net_M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_M = net_M.to(device)
    net_G = net_G.to(device)

    problem = OptimalDesign(
        batch_size=batch_size,
        net_M=net_M,
        net_G=net_G,
        n_var=n_var,
        n_obj=2,
        n_ieq_constr=3,
        zl=lower_bound,
        zu=upper_bound,
        opt_space=opt_space,
    )

    sampling = LHS(iterations=1)
    algorithm = NSGA2(pop_size=batch_size, sampling=sampling)

    res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=True, seed=42)
    print("Threads:", res.exec_time)

    plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.title("Objective Space")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()


if __name__ == "__main__":
    # main()
    batch_size = 100
    d_latent = 64
    n_layers = 4
    lr_multiplier = 0.01
    log_resolution = 6
    n_features = 16
    max_features = 64
    activation = nn.Tanh()

    net_M = MappingNetwork(d_latent, n_layers, lr_multiplier).eval()
    net_G = Generator(log_resolution, d_latent, n_features, max_features, activation=activation).eval()

    ckpt_path = r"../checkpoint/042000_133.pt"
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    net_M.load_state_dict(checkpoint["net_M_ema"])
    net_G.load_state_dict(checkpoint["net_G_ema"])

    opt_space = "w"
    if opt_space == "z" or opt_space == "w":
        blocks_zero_noise = None
        n_var = d_latent
        lower_bound, upper_bound = get_boundaries(opt_space, var_shapes=[d_latent], net_M=net_M)
    elif opt_space == "z+noise" or opt_space == "w+noise":
        blocks_zero_noise = [3, 4]
        noise = generate_noise(1, net_G.n_blocks, blocks_zero_noise=blocks_zero_noise)
        noise_shapes = [np.prod(list(elem.shape)) for i, elem in enumerate(noise) if i not in blocks_zero_noise]
        # noise_shapes[0] //= 2  # the first generator block needs only one noise
        var_shapes = [d_latent] + noise_shapes
        n_var = np.sum(var_shapes)
        lower_bound, upper_bound = get_boundaries(opt_space, var_shapes=var_shapes, net_M=net_M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_M = net_M.to(device)
    net_G = net_G.to(device)

    problem = OptimalDesign(
        batch_size=batch_size,
        net_M=net_M,
        net_G=net_G,
        n_var=n_var,
        n_obj=2,
        n_ieq_constr=3,
        zl=lower_bound,
        zu=upper_bound,
        opt_space=opt_space,
        blocks_zero_noise=blocks_zero_noise,
    )

    sampling = LHS(device=device, iterations=1000)
    algorithm = NSGA2(pop_size=batch_size, sampling=sampling)

    res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=True, seed=42)
    print("Optimization runtime:", res.exec_time)

    generate(net_M, net_G, res.X, None, opt_space, blocks_zero_noise)

    # plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors="none", edgecolors="blue")
    # plt.title("Objective Space")
    # plt.xlabel("f1")
    # plt.ylabel("f2")
    # plt.show()
