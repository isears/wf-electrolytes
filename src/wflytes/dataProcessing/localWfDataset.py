import datetime
import glob
import os
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LocalWfDataset(Dataset):
    def hadm_precheck(self):
        # Need to make sure all the assigned hadms have valid data
        valid_hadm_ids = list()

        for id in self.hadm_ids:
            for signal_name in self.signal_names:
                if os.path.exists(f"data/{id}/{signal_name}.npy"):
                    valid_hadm_ids.append(id)

        self.hadm_ids = valid_hadm_ids

    def __init__(
        self,
        signal_names: List[str],
        all_signals_required: bool = False,
        duration: datetime.timedelta = datetime.timedelta(seconds=60),
        hadm_ids: List[int] = None,
        shuffle: bool = True,
    ):
        super().__init__()

        self.signal_names = signal_names
        self.all_signals_required = all_signals_required
        self.duration = duration
        self.hadm_ids = hadm_ids

        # TODO: need to verify all records have this same sampling freq
        self.expected_fs = 125
        self.seq_len = int(self.duration.total_seconds() * self.expected_fs)
        self.min_variance = 0.1

        self.hadm_precheck()

        if shuffle:
            random.seed(42)
            random.shuffle(self.hadm_ids)

        print(f"[{type(self).__name__}] Dataset initialization complete")
        print(f"\tSignals: {self.signal_names}")
        print(f"\tWindow size: {self.duration}")

    def get_num_features(self) -> int:
        return len(self.signal_names)

    def get_seq_len(self) -> int:
        return self.seq_len

    def __len__(self) -> int:
        return len(self.hadm_ids)

    def __getitem__(self, index):
        hadm_id = self.hadm_ids[index]

        all_signals = list()

        for signal_name in self.signal_names:
            try:
                signal_data = np.load(f"data/{hadm_id}/{signal_name}.npy")
                signal_data = np.nan_to_num(signal_data, nan=0.0)
                window_size = 20  # TODO: tune this for mem / cpu tradeoff?
                stride = signal_data.strides[0]
                window_shape = (signal_data.shape[0] - window_size + 1, window_size)
                sliding_window_view = np.lib.stride_tricks.as_strided(
                    signal_data, window_shape, (stride, stride)
                )
                variances = np.var(sliding_window_view, axis=1)
                variances = variances[: signal_data.shape[0] - self.seq_len]
                valid_indices = np.where(variances > self.min_variance)[0]

                # TODO: this is terrrible and needs to be fixed by proper preprocessing of data
                # (outside the dataset object)
                if len(valid_indices) == 0:
                    signal_data = np.zeros(self.seq_len)
                else:
                    chosen_idx = np.max(valid_indices) - (self.seq_len + window_size)
                    signal_data = signal_data[chosen_idx : chosen_idx + self.seq_len]

            except FileNotFoundError as e:
                if not self.all_signals_required:
                    signal_data = np.zeros(self.seq_len)

                else:
                    raise e

            all_signals.append(signal_data)

        X = np.stack(all_signals)
        Y = np.load(f"data/{hadm_id}/label.npy")

        return torch.tensor(X), torch.tensor(Y)


if __name__ == "__main__":
    hadm_ids = [int(h.split("/")[1]) for h in glob.glob("data/*")]
    ds = LocalWfDataset(["II"], all_signals_required=True, hadm_ids=hadm_ids)

    dl = DataLoader(ds, batch_size=4, num_workers=4)

    for batchnum, (X, Y) in enumerate(dl):
        print(X)
        print(Y)
        print(batchnum)
        break
