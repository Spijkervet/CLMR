import pandas as pd
import platform
import subprocess
import os

hostname = platform.node()
experiment_log = open("experiments/{}.log".format(hostname), "a")

df = pd.read_csv("./experiments/ws.csv")
for idx, row in df.iterrows():
    # run experiments on hostname
    if hostname.lower() == row["host"]:
        config = row.to_dict()
        cmd = ["python", "main.py", "with", *["{}={}".format(k,v) for k, v in config.items()]]
        with open("cmds.txt", "a") as f:
            f.write(" ".join(cmd) + "\n\n")

experiment_log.close()
