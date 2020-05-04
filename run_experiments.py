import pandas as pd
import platform
import subprocess
import os

hostname = platform.node()
experiment_log = open(f"experiments/{hostname}.log", "a")

df = pd.read_csv("./experiments/ws7.csv")
for idx, row in df.iterrows():
    # run experiments on hostname
    if hostname == row["host"]:
        config = row.to_dict()
        cmd = ["python", "main.py", "with", *[f"{k}={v}" for k, v in config.items()]]
        with open("cmds.txt", "a") as f:
            f.write(" ".join(cmd) + "\n\n")
        # subprocess.call(cmd)
        # exit(0)
        

experiment_log.close()