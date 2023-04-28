import numpy as np
import pandas as pd
import torch
import os
import re

from wolfram import inp_preparation

def f1(x):
    result = np.sum(x, axis=(1, 2, 3)) / np.prod(x.shape[1:])
    return result


def f2(x):
    # Convert a voxel object into a finite element model
    inp_paths = inp_preparation(x, poolsize=10)
    load_path = "inp_paths.txt"
    with open(load_path, "w+") as f:
        for inp_path in inp_paths:
            f.write(inp_path + "\n")
    return 0

    # Define the file path to save the young modules and run the Abaqus
    save_path = "young_modules.csv"
    abaqus = "abaqus"
    if "ABAQUS_BAT_PATH" in os.environ.keys():
        abaqus = os.environ["ABAQUS_BAT_PATH"]
    abaqus_script_path = "util/abaqus_script.py"
    args = " ".join([load_path, save_path])
    os.system(f"{abaqus} cae noGUI={abaqus_script_path} -- {args}")

    # Load the saved young modules and return them
    young_modules = pd.read_csv(save_path, header=None, names=["inp", "young_module"])
    young_modules["inp"] = young_modules["inp"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    young_modules_dict = {key: value for key, value in zip(young_modules["inp"], young_modules["young_module"])}
    result = np.array([young_modules_dict.get(i, 0.) for i in range(len(x))])
    return result

if __name__ == "__main__":
    train_ds_path = r"C:\Users\Evgeniy\Jupyter\Work\generated_cuboids\inception_dataset\train_prepared"
    for root, folder, files, in os.walk(train_ds_path):
        if len(files) != 0:
            stats_path = os.path.join(train_ds_path, root.split("\\")[-1] + "_stats.csv")
            x = np.array([np.load(os.path.join(root, file)) for file in files]) * 0.5 + 0.5
            x = np.squeeze(x, axis=1)[:20]
            print(len(x))
            volume_fraction = f1(x)
            young_modules = f2(x)
            break

        print(root)
        print(folder)
        print(files)