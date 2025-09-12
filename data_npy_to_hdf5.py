"""
Create an HDF5 file from multiple NumPy data files.

This script reads `.npy` files from a specified folder and creates HDF5 datasets
with the following keys:

    keys = [
        'Eout_bti', 'Eout_strongLandau', 'Eout_tsi', 'Eout_weakLandau',
        'pos_bti', 'pos_strongLandau', 'pos_tsi', 'pos_weakLandau'
    ]

If multiple files match the same key, their data are **stacked along the first axis**
into the same dataset. Files can have additional suffixes or extensions.

Usage:
    python data_npy_to_hdf5.py --picFolder picData --hdf5File dataset.h5

Arguments:
    --picFolder   Path to the folder containing the `.npy` files.
    --hdf5File    Path to the output HDF5 file to create or append to.

"""


import os
import h5py
import glob
import numpy as np
import argparse
# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Create HDF5 dataset from PIC simulation Numpy files',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--picFolder", default="picData", help="Folder containing numpy files")
parser.add_argument(
    "--hdf5File", default="dataset.h5", help="name of the dataset HDF5 file")
args = parser.parse_args()

npy_folder = args.picFolder
npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

keys = [
    'Eout_bti', 'Eout_strongLandau', 'Eout_tsi', 'Eout_weakLandau',
    'pos_bti', 'pos_strongLandau', 'pos_tsi', 'pos_weakLandau'
]


with h5py.File(args.hdf5File, "a") as h5f:  
    for key in keys:
        # Find all files that contain the key
        matched_files = [f for f in npy_files if key in os.path.basename(f)]
        if not matched_files:
            print(f"Warning: No .npy file found for key '{key}'")
            continue

        for npy_file in matched_files:
            data = np.load(npy_file)
            data = np.atleast_2d(data)  # ensure 2D for vertical stacking

            if key in h5f:
                # Append to existing dataset
                dset = h5f[key]
                old_len = dset.shape[0]
                new_len = old_len + data.shape[0]
                dset.resize((new_len, *dset.shape[1:]))
                dset[old_len:new_len] = data
                print(f"Appended {data.shape[0]} entries to key '{key}' from {os.path.basename(npy_file)}")
            else:
                # Create resizable dataset
                maxshape = (None, *data.shape[1:])  # None means unlimited along first axis
                chunks = (min(1000, data.shape[0]), *data.shape[1:])  # chunking 
                h5f.create_dataset(key, data=data, maxshape=maxshape)
                print(f"Created key '{key}' from {os.path.basename(npy_file)}")

