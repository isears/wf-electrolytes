import glob
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Generator

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
    """
    Designed to be lazy for minimal memory footprint when used with dataset
    """
    hadm_id: int
    instantation_record: wfdb.Record = None

    def __post_init__(self):
        self._path = f"data/{self.hadm_id}"
        assert os.path.exists(self._path)

    def _load_metadata(self) -> dict:
        with open(f"{self._path}/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    def update_metadata(self, updates: dict[any, any]) -> None:
        metadata = self._load_metadata()

        for k, v in updates.items():
            metadata[k] = v

        with open(f"{self._path}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    def get_signal_names(self) -> list[str]:
        return self._load_metadata()['sig_name']

    def generate_signals(self) -> Generator[tuple[str, np.ndarray], None, None]:
        for signal_name in self.get_signal_names():
            yield signal_name, np.load(f"{self._path}/{signal_name}.npy")

    def has_signal(self, signal_name: str) -> bool:
        return signal_name in self._load_metadata()['sig_name']

    def get_full_signal(self, signal_name: str) -> np.ndarray:
        if not os.path.exists(f"{self._path}/{signal_name}.npy"):
            raise ValueError(f"{signal_name} signal not found for hadm id {self.hadm_id}")

        return np.load(f"{self._path}/{signal_name}.npy")

    def get_segment_count(self, signal_name: str) -> int:
        if not os.path.exists(f"{self._path}/processed/{signal_name}"):
            return 0
        
        return len(glob.glob(f"{self._path}/processed/{signal_name}/*.npy"))

    def get_segment(self, signal_name: str, idx: int) -> np.ndarray:
        all_segments = glob.glob(f"{self._path}/processed/{signal_name}/*.npy")
        return np.load(all_segments[idx])

    def cache_segment(self, signal_name: str, signal: np.ndarray, start_idx: int, end_idx: int):
        if not os.path.exists(f"{self._path}/processed/{signal_name}"):
            os.makedirs(f"{self._path}/processed/{signal_name}")

        segment = signal[start_idx: end_idx]
        np.save(f"{self._path}/processed/{signal_name}/{start_idx}_{end_idx}", segment)

    def clean_processed_cache(self) -> None:
        if os.path.exists(f"{self._path}/processed"):
            assert ".." not in self._path, "[-] Uhhhhhh...."
            shutil.rmtree(f"{self._path}/processed")

        os.makedirs(f"{self._path}/processed")
        
    def get_signals(self, signal_names: list[str]) -> np.ndarray:
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
