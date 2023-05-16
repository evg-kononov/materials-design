import pandas as pd
import re
import os
# Convert a voxel object into a finite element model
#save_path = r"C:\Users\Evgeniy\Jupyter\Work\materials-design\optimization\util\inps"
save_path = r"D:\inception_dataset\inps_64_uniform_1"
inp_paths = [os.path.join(os.path.abspath(save_path), inp) for inp in os.listdir(save_path)]
load_path = "inp_paths.txt"
with open(load_path, "w+") as f:
    for inp_path in inp_paths:
        f.write(inp_path + "\n")

if os.path.exists("young_modules.csv"):
    os.remove("young_modules.csv")
if os.path.exists("sys_exit.txt"):
    os.remove("sys_exit.txt")
start_idx = 0
while True:
    if os.path.exists("sys_exit.txt"):
        with open("sys_exit.txt", "r+") as f:
            start_idx = f.readline()
            print(start_idx)
        if start_idx == "end":
            break
        else:
            start_idx = int(start_idx) + 1
    # Define the file path to save the young modules and run the Abaqus
    save_path = "young_modules.csv"
    abaqus = "abaqus"
    if "ABAQUS_BAT_PATH" in os.environ.keys():
        abaqus = os.environ["ABAQUS_BAT_PATH"]
    abaqus_script_path = "abaqus_script.py"
    args = " ".join([str(start_idx), load_path, save_path])
    os.system(f"{abaqus} cae noGUI={abaqus_script_path} -- {args}")



print("GOGOGO")