"""
Need to take variable-length and potentially v. long sequences and divide up
into more manageable pieces
"""
import datetime
import glob
import multiprocessing

import numpy as np
from tqdm import tqdm

from wflytes.dataProcessing import HadmWfRecord


def handle_single_hadm(
    hadm_id: int,
    duration: datetime.timedelta = datetime.timedelta(seconds=60),
    min_variance: float = 0.1,
):
    record = HadmWfRecord(hadm_id)
    record.clean_processed_cache()
    metadata = record._load_metadata()

    desired_length = int(metadata["fs"] * duration.total_seconds())

    for sig_name, sig in record.generate_signals():
        sig = np.nan_to_num(sig, nan=0.0)

        # pad array up to size that's divisible by desired_length
        pad_size = int(np.ceil(sig.size / desired_length) * desired_length) - sig.size
        sig = np.pad(sig, (0, pad_size))

        # TODO: sliding window may get us more data, but is much more time-consuming
        sig = sig.reshape(int(sig.size / desired_length), desired_length)

        segment_variances = np.var(sig, axis=1)

        for idx, var in enumerate(segment_variances):
            if var > min_variance:
                record.cache_segment(
                    sig_name, sig, idx * desired_length, idx + desired_length
                )

    record.update_metadata({"duration": duration})


if __name__ == "__main__":

    for dirname in tqdm(glob.glob("data/*")):
        hadm_id = int(dirname.split("/")[-1])
        handle_single_hadm(hadm_id, datetime.timedelta(seconds=60))
