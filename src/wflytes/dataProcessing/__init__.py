import os
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import wfdb

potassium_events = [
    # 227442,  # Serum, metavision (total count 165,813)
    # 227464,  # Whole blood, metavision (total count 49,112)
    # 220640,  # ZPotassium, serum, metavision (total count 0)
    # 226535,  # ZPotassium, whole blood, metavision (total count 0)
    1535,  # Carevue (total count 246,969)
]


@dataclass
class HadmWfRecord:
    hadm_id: int
    instantation_record: wfdb.Record = None

    def __post_init__(self):
        self._path = f"data/{self.hadm_id}"
        assert os.path.exists(self._path)

    def has_signal(self, signal_name: str) -> bool:
        raise NotImplementedError()

    def get_signal(self, signal_name: str) -> np.ndarray:
        raise NotImplementedError()

    def get_signals(self, signal_names: List[str]) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def is_valid(hadm_id: int) -> bool:
        return os.path.exists(f"data/{hadm_id}")


def wfdb_record_to_hadmwf_record(hadm_id: int, record: wfdb.Record) -> HadmWfRecord:
    """
    Creates the underlying directory structure necessary for a HadmWfRecord
    """
    os.makedirs(f"data/{hadm_id}", exist_ok=True)

    metadata = {k: v for k, v in record.__dict__.items() if k != "p_signal"}

    with open(f"data/{hadm_id}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    for idx, sig_name in enumerate(record.sig_name):
        sig_data = record.p_signal[:, idx]
        np.save(f"data/{hadm_id}/{sig_name}", sig_data)

    return HadmWfRecord(hadm_id)
