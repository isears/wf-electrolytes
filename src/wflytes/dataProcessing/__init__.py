import glob
import os
import pickle
import random
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
        with open(f"{self._path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return metadata

    def update_metadata(self, updates: dict[any, any]) -> None:
        metadata = self._load_metadata()

        for k, v in updates.items():
            metadata[k] = v

        with open(f"{self._path}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def get_signal_names(self) -> list[str]:
        return self._load_metadata()["sig_name"]

    def generate_signals(self) -> Generator[tuple[str, np.ndarray], None, None]:
        for signal_name in self.get_signal_names():
            yield signal_name, np.load(f"{self._path}/{signal_name}.npy")

    def has_signal(self, signal_name: str) -> bool:
        return signal_name in self._load_metadata()["sig_name"]

    def get_full_signal(self, signal_name: str) -> np.ndarray:
        if not os.path.exists(f"{self._path}/{signal_name}.npy"):
            raise ValueError(
                f"{signal_name} signal not found for hadm id {self.hadm_id}"
            )

        return np.load(f"{self._path}/{signal_name}.npy")

    def get_segment_count(self, signal_name: str) -> int:
        if not os.path.exists(f"{self._path}/processed/{signal_name}"):
            return 0

        return len(glob.glob(f"{self._path}/processed/{signal_name}/*.npy"))

    def get_overlapping_segment_count(self, signal_names: list[str]) -> int:
        if len(signal_names) == 0:
            return 0
        elif len(signal_names) == 1:
            return self.get_segment_count(signal_names[0])

        for name in signal_names:
            if self.get_segment_count(name) == 0:
                return 0

        overlap_count = 0
        fnames = [os.listdir(f"{self._path}/processed/{n}/") for n in signal_names]
        first_signal_fnames = fnames[0]
        other_signal_fnames = fnames[1:]

        for fname in first_signal_fnames:
            if all([fname in i for i in other_signal_fnames]):
                overlap_count += 1

        return overlap_count

    def get_segment(self, signal_name: str, idx: int) -> np.ndarray:
        all_segments = glob.glob(f"{self._path}/processed/{signal_name}/*.npy")
        return np.load(all_segments[idx])

    def cache_segment(
        self, signal_name: str, signal: np.ndarray, start_idx: int, end_idx: int
    ):
        if not os.path.exists(f"{self._path}/processed/{signal_name}"):
            os.makedirs(f"{self._path}/processed/{signal_name}")

        segment = signal[start_idx:end_idx]
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


@dataclass
class HadmWfOverlappingSignalRecord:
    hadm_id: int
    signal_names: list[str]

    def __post_init__(self):
        assert len(self.signal_names) > 0

        self.record = HadmWfRecord(self.hadm_id)

        if len(self.signal_names) == 1:
            try:
                self.fnames = os.listdir(
                    f"{self.record._path}/processed/{self.signal_names[0]}/"
                )
            except FileNotFoundError:
                self.fnames = []

        else:
            for name in self.signal_names:
                if self.get_segment_count(name) == 0:
                    return 0

                try:
                    fnames = [
                        os.listdir(f"{self.record._path}/processed/{n}/")
                        for n in self.signal_names
                    ]
                except FileNotFoundError:
                    fnames = list()

                first_signal_fnames = fnames[0]
                other_signal_fnames = fnames[1:]

                self.fnames = list()

                for fname in first_signal_fnames:
                    if all([fname in i for i in other_signal_fnames]):
                        self.fnames.append(fname)

    def get(self, idx: int):
        ret = list()
        for n in self.signal_names:
            ret.append(np.load(f"{self.record._path}/processed/{n}/{self.fnames[idx]}"))

        return np.stack(ret)

    def howmany(self) -> int:
        return len(self.fnames)


@dataclass
class HadmWfRecordCollection:
    hadm_ids: list[int]
    signal_names: list[str]
    shuffle: bool = True

    def __post_init__(self):
        self.records = [
            HadmWfOverlappingSignalRecord(h, self.signal_names) for h in self.hadm_ids
        ]

        self.indexer = list()
        for r in self.records:
            for idx in range(0, r.howmany()):
                self.indexer.append((r, idx))

        if self.shuffle:
            random.seed(42)
            random.shuffle(self.indexer)

    def howmany(self) -> int:
        return len(self.indexer)

    def get_X(self, idx: int) -> np.ndarray:
        record, jdx = self.indexer[idx]
        return record.get(jdx)

    def get_Y(self, idx: int) -> np.ndarray:
        record, jdx = self.indexer[idx]
        return np.load(f"{record.record._path}/label.npy")


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
