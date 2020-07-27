import csv
import requests
import socket

def csv_to_dict(content):
    reader = csv.reader(content.splitlines(), delimiter=',')
    header = next(reader)
    experiments = {}
    for row in reader:
        d = {}
        i = row[0]
        for idx, r in enumerate(row):
            d[header[idx]] = r
        experiments[i] = d
    return experiments

def get_host_experiment(experiments):
    num_exp = 0
    exp = None
    for exec_id, params in experiments.items():
        if params["host"] == socket.gethostname() and params["status"] == "pending":
            exp = experiments[exec_id]
            num_exp += 1
    assert num_exp < 2, "There are multiple experiments defined for this machine. Set status to 'running' to start a new one on the same machine."
    return exp

with requests.Session() as s:
    r = s.get('https://docs.google.com/spreadsheet/ccc?key=1iSKcoSqA7tOKcsuaKgV_Bz57V_2yZ_XXe9nn1-ApCEA&output=csv')
    assert r.status_code == 200, 'Wrong status code'
    decoded = r.content.decode("utf-8")
    experiments = csv_to_dict(decoded)
    host_experiment = get_host_experiment(experiments)
    
    gpu_nr = host_experiment["gpu_nr"]
    # delete unused keys
    del host_experiment["gpu_nr"]
    del host_experiment["status"]
    del host_experiment["host"]
    ks = []
    for k in host_experiment.keys():
        if "auc" in k:
            ks.append(k)
    for k in ks:
        del host_experiment[k]
        
    cmd = "CUDA_VISIBLE_DEVICES={} python main.py ".format(gpu_nr)
    cmd += " ".join(["--{} {}".format(k, v) for k, v in host_experiment.items()])

    print("Copy/paste the following command to start training:")
    print()
    print(cmd)
    print()