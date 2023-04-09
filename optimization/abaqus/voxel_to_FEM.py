import time
import asyncio
import shutil
import numpy as np
from wolframclient.evaluation import WolframEvaluatorPool
from wolframclient.language import wl

from expressions import expr


async def main():
    async with WolframEvaluatorPool(poolsize=poolsize) as pool:
        start = time.perf_counter()

        tasks = [pool.evaluate(expr) for i in range(len(x))]
        await asyncio.wait(tasks)
        tasks = []
        for i, x_i in enumerate(x):
            # try:
            #     task = pool.evaluate(wl.Global.toFEM(x_i, i))
            # except:
            #     pass
            task = pool.evaluate(wl.Global.toFEM(x_i, i))
            tasks.append(task)
        await asyncio.wait(tasks)

        print('Done after %.02fs, using up to %i kernels.'
              % (time.perf_counter() - start, len(pool)))


if __name__ == "__main__":
    poolsize = 5
    x = np.load("val_cuboids_normal.npy")[1:12:2]
    # x = np.random.randint(low=0, high=2, size=(4, 32, 32, 32))
    x = x.astype(np.float32)

    asyncio.run(main())

    # try:
    #     shutil.rmtree("samples")
    # except IOError as ex:
    #     print(ex)
