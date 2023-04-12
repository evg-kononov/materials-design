import os
import time
import asyncio
import shutil
import numpy as np

from wolframclient.evaluation import WolframEvaluatorPool
from wolframclient.language import wl
from expressions import expr


async def main(x, poolsize):
    async with WolframEvaluatorPool(poolsize=poolsize) as pool:
        start = time.perf_counter()

        tasks = [pool.evaluate(expr) for i in range(len(x))]
        await asyncio.wait(tasks)
        tasks = []
        for i, x_i in enumerate(x):
            task = pool.evaluate(wl.Global.toFEM(x_i, i))
            tasks.append(task)
        await asyncio.wait(tasks)

        print("Done inp creation after %.02fs, using up to %i kernels."
              % (time.perf_counter() - start, len(pool)))


def inp_preparation(x, poolsize, save_path="inps"):
    try:
        os.mkdir(save_path)
    except IOError as ex:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        print(ex)

    asyncio.run(main(x, poolsize))
    return [os.path.join(save_path, inp) for inp in os.listdir(save_path)]


if __name__ == "__main__":
    print("WOLFRAM TEST")
    poolsize = 2
    # x = np.load("val_cuboids_normal.npy")[:10]
    x = np.random.randint(low=0, high=2, size=(4, 32, 32, 32))
    x = x.astype(np.float32)

    inp_preparation(x, poolsize)
