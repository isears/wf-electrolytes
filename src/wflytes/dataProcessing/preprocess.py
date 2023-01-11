"""
Need to take variable-length and potentially v. long sequences and divide up
into more manageable pieces
"""
import datetime
import glob
import os
import pickle
import shutil

import numpy as np


def handle_single_dir(dirname, duration: datetime.timedelta, min_variance: float = 0.1):
    if os.path.exists(f"{dirname}/processed"):
        assert ".." not in dirname, "[-] Uhhhhhh...."
        shutil.rmtree(f"{dirname}/processed")

    os.makedirs(f"{dirname}/processed")

    with open(f"{dirname}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    desired_length = int(metadata["fs"] * duration.total_seconds())

    for sig_file in glob.glob(f"{dirname}/*.npy"):
        sig = np.load(sig_file)

        window_size = 20  # Can't do full desired_length b/c it will run out of mem
        stride = sig.strides[0]
        window_shape = (sig.shape[0] - window_size + 1, window_size)
        sliding_window_view = np.lib.stride_tricks.as_strided(
            sig, window_shape, (stride, stride)
        )
        variances = np.var(sliding_window_view, axis=1)
        variances = variances[: sig.shape[0] - desired_length]
        valid_indices = np.where(variances > min_variance)[0]

        print("deleteme")


if __name__ == "__main__":
    handle_single_dir("data/189051", datetime.timedelta(seconds=60))
