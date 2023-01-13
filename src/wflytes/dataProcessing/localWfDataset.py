import datetime
import glob
import os
import random
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from wflytes.dataProcessing import HadmWfRecordCollection


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

        self.collection = HadmWfRecordCollection(self.hadm_ids, self.signal_names)

        print(f"[{type(self).__name__}] Dataset initialization complete")
        print(f"\tSignals: {self.signal_names}")
        print(f"\tWindow size: {self.duration}")

    def get_num_features(self) -> int:
        return len(self.signal_names)

    def get_seq_len(self) -> int:
        return self.seq_len

    def collate_skorch(self, batch):
        """
        Skorch expects kwargs output
        """
        X = torch.stack([X[20:1000] for X, _ in batch], dim=0)
        y = torch.stack([Y for _, Y in batch], dim=0)
        pad_mask = torch.stack(
            [torch.ones(X.shape[1]).bool() for idx in range(0, len(batch))], dim=0
        )

        return dict(X=X, padding_masks=pad_mask), torch.squeeze(y, dim=1)

    def __len__(self) -> int:
        return self.collection.howmany()

    def __getitem__(self, index):
        X = torch.tensor(self.collection.get_X(index))
        Y = torch.tensor(self.collection.get_Y(index))

        return X.float(), Y.float()


if __name__ == "__main__":
    hadm_ids = [int(h.split("/")[1]) for h in glob.glob("data/*")]
    ds = LocalWfDataset(["II"], all_signals_required=True, hadm_ids=hadm_ids)

    dl = DataLoader(ds, batch_size=4, num_workers=4)

    for batchnum, (X, Y, pm) in enumerate(dl):
        print(X)
        print(Y)
        print(batchnum)
        break
