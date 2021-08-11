import os
import sys
import json
import random
import logging
from typing import Dict, Union, Optional

import torch
import numpy as np
from types import SimpleNamespace

from utils.train import EMA
from datasets.datasets import CustomSubset


logger = logging.getLogger()


def initialize_logger(save_path: str):
    logger = logging.getLogger()
    logging.basicConfig(
        filename=os.path.join(save_path, "log.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def seed(random_seed: bool, seed: int):
    seed = random.randint(0, 100) if random_seed else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_save_path(args):
    save_dir = os.path.join(args.out_dir, "fixmatch_training")
    os.makedirs(save_dir, exist_ok=True)
    num_existing_dirs = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, "run_{}".format(num_existing_dirs))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_args(args, save_path: str):
    args_dict = vars(args)
    args_file_path = os.path.join(save_path, "args.json")
    with open(args_file_path, "w") as file:
        json.dump(args_dict, file)


def save_dataset_indices(
        save_path: str,
        train_set: Union[CustomSubset, Dict],
        validation_set: CustomSubset,
        file_name: str = "indices.json",
):
    label_file_path = os.path.join(save_path, file_name)
    dataset_indices = {}
    if isinstance(train_set, Dict):
        for subset_name, subset in train_set.items():
            dataset_indices["train_{}".format(subset_name)] = list(
                map(lambda x: int(x), subset.indices)
            )
    else:
        dataset_indices["train_labeled"] = list(
            map(lambda x: int(x), train_set.indices)
        )
    dataset_indices["validation"] = list(map(lambda x: int(x), validation_set.indices))
    with open(label_file_path, "w") as file:
        json.dump(dataset_indices, file)


def save_state(
        epoch: int,
        model: torch.nn.Module,
        val_acc: float,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema_model: Optional[EMA],
        path: str,
        filename: str = "best_model.tar",
):
    old_checkpoint_files = list(
        filter(lambda x: "checkpoint" in x, os.listdir(path))
    )

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "acc": val_acc,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "ema_model_shadow": None if ema_model is None else ema_model.shadow,
    }
    file_path = os.path.join(path, filename)
    logger.info("Save current state to {}".format(filename))
    torch.save(state_dict, file_path)

    for file in old_checkpoint_files:
        os.remove(os.path.join(path, file))


def load_dataset_indices(load_path: str, file_name: str = "indices.json"):
    with open(os.path.join(load_path, file_name), "r") as file:
        indices = json.load(file)
    return indices


def load_state(path: str, map_location=None):
    loaded_state = torch.load(path, map_location=map_location)
    logger.info(
        "Loaded state from {} saved at epoch {}".format(path, loaded_state["epoch"])
    )
    return loaded_state


def load_args(run_path):
    run_args = json.load(open(os.path.join(run_path, "args.json")))
    return SimpleNamespace(**run_args)
