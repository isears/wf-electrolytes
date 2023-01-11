"""
Write a label numpy array to each directory that contains data indicating whether or not stroke occured
"""
import glob
import os

import numpy as np
import pandas as pd

from wflytes.dataProcessing.downloadStrokeWf import get_stroke_hdms

if __name__ == "__main__":
    stroke_hadms = get_stroke_hdms()

    pos_count = 0
    neg_count = 0
    unlabeled_count = 0

    for subdir in glob.glob("data/*"):
        if os.path.isdir(subdir):
            hadm_id = int(subdir.split("/")[-1])

            if hadm_id in stroke_hadms:
                label = np.array([1.0])
                pos_count += 1
            else:
                label = np.array([0.0])
                neg_count += 1

            np.save(f"{subdir}/label", label)

    print(f"Labeled {pos_count + neg_count} records")
    print(f"\t{pos_count} positive")
    print(f"\t{neg_count} negative")
    print(f"\t{unlabeled_count} unlabeled")
