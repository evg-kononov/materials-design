import torch
import numpy as np

import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def g(x):
    return x


def f1(x, n=64):
    return np.sum(g(x), axis=1) / n ** 3


def f2(x):
    return np.sum(g(x), axis=1) - np.sum(g(x), axis=1) ** 2


class OptimalDesign(Problem):
    def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        g1 = -f1(x)
        g2 = f1(x) - 1
        g3 = -f2(x)

        out["F"] = [f1(x), f2(x)]
        out["G"] = [g1, g2, g3]


if __name__ == "__main__":
    d_latent = 64
    xl, xu = -5, 5

    problem = OptimalDesign(n_var=d_latent, n_obj=2, n_ieq_constr=3, xl=xl, xu=xu)

    sampling = LHS()
    algorithm = NSGA2(pop_size=3000, sampling=sampling)

    res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=True, seed=42)
    print("Threads:", res.exec_time)

    plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.title("Objective Space")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()
