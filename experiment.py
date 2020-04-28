"""
Sacred experiment file
"""
from pathlib import Path

# Sacred
from sacred import Experiment
from sacred.stflow import LogFileWriter
from sacred.observers import FileStorageObserver, MongoObserver

from utils import CustomFileStorageObserver

# custom config hook
from utils.yaml_config_hook import yaml_config_hook


ex = Experiment("CLMR")

#### database output
# ex.observers.append(
#     MongoObserver().create(
#         url=f"mongodb://admin:admin@localhost:27017/?authMechanism=SCRAM-SHA-1",
#         db_name="db",
#     )
# )


@ex.config
def my_config():
    # config_file = "./config/config_audio_fma_16000.yaml"
    config_file = "./config/config_audio_billboard_16000.yaml"
    # config_file = "./config/config_audio_magnatagatune_16000.yaml"

    ex.add_config(config_file)

    cfg = yaml_config_hook(config_file)
    ex.add_config(cfg)


    #### file output directory
    ex.observers.append(FileStorageObserver(Path("./logs", cfg["domain"], cfg["dataset"], cfg["task"])))
    del cfg


    # override any settings here
    # start_epoch = 100
    # ex.add_config(
    #   {'start_epoch': start_epoch})
