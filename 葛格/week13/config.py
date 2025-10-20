# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "/home/usr3/pycharm_projects/self_project/homework/output",
    "train_data_path": "/home/usr3/pycharm_projects/self_project/homework/train_data.txt",
    "valid_data_path": "/home/usr3/pycharm_projects/self_project/homework/valid_data.txt",
    "vocab_path":"/home/usr3/pycharm_projects/self_project/homework/chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "pooling_style":"avg",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":"/home/usr3/pycharm_projects/self_project/bert-base-chinese",
    "seed": 987
}

