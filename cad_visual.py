import argparse
from visualize.visualize_cad import visualize_multiple

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    default="trained_results/cad/best.pth",
    type=str,
    help="Path to ckpt path",
    dest="checkpoint_path",
)
parser.add_argument(
    "--valid",
    default="cad.h5",
    type=str,
    help="Path to a file containing valid data",
)
parser.add_argument(
    "--data",
    default="data/cad",
    type=str,
    help="Path to the folder containing the data",
    dest="data_path",
)
parser.add_argument(
    "--data_type",
    default="cad",
    type=str,
    choices=["synthetic", "cad"],
    help="Data to be used to reconstruc images",
)

args = parser.parse_args()
visualize_multiple(**vars(args))