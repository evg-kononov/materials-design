import os
import time
import asyncio
import shutil
import numpy as np

from wolframclient.evaluation import WolframEvaluatorPool
from wolframclient.evaluation.kernel.kernelcontroller import WolframKernelController
from wolframclient.language import wl
from expressions import expr

import logging

# set the Python root logger level to INFO
logging.basicConfig(level=logging.INFO)

async def main(x, idxs, poolsize):
    async with WolframEvaluatorPool(
        poolsize=poolsize,
        kernel_loglevel=logging.INFO,

    ) as pool:
        start = time.perf_counter()

        tasks = [pool.evaluate(expr) for i in range(len(x))]

        await asyncio.gather(*tasks)
        tasks = []
        for x_i, i in zip(x, idxs):
            task = pool.evaluate(wl.Global.toFEM(x_i, i))
            tasks.append(task)

        await asyncio.gather(*tasks)

        print("Done inp creation after %.02fs, using up to %i kernels."
              % (time.perf_counter() - start, len(pool)))


def inp_preparation(x, poolsize, save_path="inps"):
    try:
        os.mkdir(save_path)
    except IOError as ex:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
        print(ex)

    numbers = np.arange(0, len(x))
    for slice, idxs in zip(np.array_split(x, len(x) // poolsize), np.array_split(numbers, len(numbers) // poolsize)):
        try:
            print(idxs)
            asyncio.run(main(slice, idxs, poolsize))
        except Exception as ex:
            print(ex)
            pass
    return [os.path.join(os.path.abspath(save_path), inp) for inp in os.listdir(save_path)]


def f2(x):
    import pandas as pd
    import re
    import os
    # Convert a voxel object into a finite element model
    #inp_paths = inp_preparation(x, poolsize=20)
    #save_path = r"C:\Users\Evgeniy\Jupyter\Work\materials-design\optimization\util\inps"
    save_path = r"D:\inception_dataset\inps_64_normal_0"
    inp_paths = [os.path.join(os.path.abspath(save_path), inp) for inp in os.listdir(save_path)]
    load_path = "inp_paths.txt"
    with open(load_path, "w+") as f:
        for inp_path in inp_paths:
            f.write(inp_path + "\n")

    # Define the file path to save the young modules and run the Abaqus
    try:
        save_path = "young_modules.csv"
        abaqus = "abaqus"
        if "ABAQUS_BAT_PATH" in os.environ.keys():
            abaqus = os.environ["ABAQUS_BAT_PATH"]
        abaqus_script_path = "abaqus_script.py"
        args = " ".join([load_path, save_path])
        os.system(f"{abaqus} cae script={abaqus_script_path} -- {args}")
    except Exception as ex:
        print(ex)

    # Load the saved young modules and return them
    young_modules = pd.read_csv(save_path, header=None, names=["inp", "young_module"])
    young_modules["inp"] = young_modules["inp"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    young_modules_dict = {key: value for key, value in zip(young_modules["inp"], young_modules["young_module"])}
    result = np.array([young_modules_dict.get(i, 0.) for i in range(len(x))])
    return result


if __name__ == "__main__":
    print("WOLFRAM TEST")
    poolsize = 12
    #x = np.load("val_cuboids_normal.npy")[:20]
    # x = np.random.randint(low=0, high=2, size=(4, 32, 32, 32))
    path = r"C:\Users\Evgeniy\Jupyter\Work\generated_cuboids\inception_dataset\train_prepared\64_spheresRVE"
    x = [np.load(os.path.join(path, file)) for file in sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))]
    x = np.array(x)
    x = np.squeeze(x, axis=1)
    x = x * 0.5 + 0.5
    x = x.astype(np.float32)

    #inp_preparation(x, poolsize)
