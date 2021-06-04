'''
Copied from https://github.com/kacperkan/ucsgnet
'''

import torch
import torch.nn as nn

FLOAT_EPS = torch.finfo(torch.float32).eps
RNN_LATENT_SIZE = 256


class Evaluation3D:
    CATEGORY_LIST = [
        "02691156_airplane",
        "02828884_bench",
        "02933112_cabinet",
        "02958343_car",
        "03001627_chair",
        "03211117_display",
        "03636649_lamp",
        "03691459_speaker",
        "04090263_rifle",
        "04256520_couch",
        "04379243_table",
        "04401088_phone",
        "04530566_vessel",
    ]
    CATEGORY_IDS = [name[:8] for name in CATEGORY_LIST]
    CATEGORY_NAMES = [name.split("_")[1] for name in CATEGORY_LIST]
    NUM_POINTS = 4096

    CATEGORY_COUNTS = [
        809,
        364,
        315,
        1500,
        1356,
        219,
        464,
        324,
        475,
        635,
        1702,
        211,
        388,
    ]
