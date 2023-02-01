from src.content_tower import MHAContentTower
from src.user_tower import DummyUserTower, MHAUserTower
from src.utils import TimeDistributed, get_model_dir
from src.trainer import Trainer

import os

import torch
from transformers import BertConfig

import argparse

if __name__ == "__main__":

    #defaults
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    save_dir = get_model_dir()

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--save_dir", type=str, default=save_dir)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # load config and other values
    save_dir = args.save_dir
    config_path = args.config_path
    config = BertConfig.from_json_file(config_path)

    # create model
    content_tower = MHAContentTower()
    content_tower.init_from_config(config)
    time_distributed_content_tower = TimeDistributed(content_tower) # batch first?
    #user_tower = DummyUserTower(time_distributed_content_tower)
    user_tower = MHAUserTower(config=config)
    trainer : Trainer = Trainer(time_distributed_content_tower, user_tower, config=config)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.set_device(device)

    # train
    trainer.fit(args.epochs)

    # save
    trainer.save(
        save_dir
    )


