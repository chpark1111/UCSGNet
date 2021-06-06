import argparse
from visualize.visualize_3d import reconstruct_3d

parser = argparse.ArgumentParser(
    description=(
        "Reconstructs all shapes in the dataset by predicting values at "
        "each 3D point and then thresholding"
    ),
    add_help=False,
)

parser.add_argument(
    "--weights_path", 
    default="trained_results/3d/3d_best.pth", 
    help="Path to the model to load"
)
parser.add_argument(
    "--size", 
    type=int, 
    help="Data size to be used", 
    default=64
)
parser.add_argument(
    "--processed",
    dest="processed_data_path",
    type=str,
    help="Base folder of processed data",
    default="data/hdf5/"
)
parser.add_argument(
    "--valid",
    dest="valid_file",
    type=str,
    help="Path to valid HDF5 file with the valid data",
    default="data/hdf5/all_vox256_img_test.hdf5"
)
parser.add_argument(
    "--valid_shape_names",
    type=str,
    help=(
        "Path to valid text file with the names for each data point in "
        "the valid dataset"
    ),
    default="all_vox256_img_test.txt"
)
parser.add_argument(
    "--sphere_complexity",
    type=int,
    help="Number of segments lat/lon of the sphere",
    default=2,
)
args = parser.parse_args()
reconstruct_3d(args)