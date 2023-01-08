import datetime
import random

import numpy as np
import pandas as pd
import tqdm
import wfdb


class WfDataNotFoundError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class WfFetcher:
    root_dir = "mimic3wdb-matched/1.0"

    def __init__(self, min_variance: float = 0.1) -> None:
        self.min_variance = min_variance
        random.seed(42)

        self.admissions = pd.read_csv("mimiciii/ADMISSIONS.csv")

        for t_col in ["ADMITTIME", "DISCHTIME"]:
            self.admissions[t_col] = pd.to_datetime(self.admissions[t_col])

    def _hadm_to_subject(self, hadm_id: int) -> int:
        possible_subjects = self.admissions[self.admissions["HADM_ID"] == hadm_id][
            "SUBJECT_ID"
        ].to_list()

        assert (
            len(possible_subjects) > 0
        ), "[-] Error, couldn't find any subjects to match the hadm"
        assert (
            len(possible_subjects) < 2
        ), "[-] Error, multiple subjects returned for hadm"

        return possible_subjects[0]

    def _subject_to_hadm(self, subject_id: int, date: datetime.datetime) -> int:
        possible_hadms = self.admissions[
            (date >= self.admissions["ADMITTIME"])
            & (date < self.admissions["DISCHTIME"])
            & (self.admissions["SUBJECT_ID"] == subject_id)
        ]

        if len(possible_hadms) == 0:
            # Sometimes there are subject id / datetime combinations in the wfdb
            # that don't appear to match any hadm
            return None

        assert (
            len(possible_hadms) < 2
        ), "[-] Error, multiple hadms returned for subject / date"

        return possible_hadms["HADM_ID"].iloc[0]

    def _sid_to_dir(self, sid: int) -> str:
        zfilled = str(sid).zfill(6)
        return f"{self.root_dir}/p{zfilled[0:2]}/p{zfilled}/"

    def _get_valid_signal(
        self, signal_in: np.ndarray, desired_length: int
    ) -> np.ndarray:
        """
        Get signal that's not nan and has some degree of variance
        """
        assert signal_in.ndim == 1, "Pass one signal at a time to this fn"

        signal_in = np.nan_to_num(signal_in, nan=0.0)
        window_size = 20  # Can't do full desired_length b/c it will run out of mem
        stride = signal_in.strides[0]
        window_shape = (signal_in.shape[0] - window_size + 1, window_size)
        sliding_window_view = np.lib.stride_tricks.as_strided(
            signal_in, window_shape, (stride, stride)
        )
        variances = np.var(sliding_window_view, axis=1)
        variances = variances[: signal_in.shape[0] - desired_length]
        valid_indices = np.where(variances > self.min_variance)[0]

        if len(valid_indices) == 0:
            raise WfDataNotFoundError(
                f"Record exists, but couldn't find signal with variance > {self.min_variance}"
            )

        chosen_idx = random.choice(valid_indices)

        return signal_in[chosen_idx : chosen_idx + desired_length]

    def _get_record_by_hadm(self, hadm_id: int, sig_name: str) -> wfdb.Record:
        subject_id = self._hadm_to_subject(hadm_id)
        dirname = self._sid_to_dir(subject_id)
        try:
            record_list = wfdb.get_record_list(dirname)
        except ValueError:
            raise WfDataNotFoundError(
                f"No record found whatsoever for hadm id: {hadm_id}"
            )

        dated_records = [r for r in record_list if "-" in r and "n" not in r]

        for r in dated_records:
            header = wfdb.io.rdheader(r, pn_dir=dirname)

            if hadm_id == self._subject_to_hadm(subject_id, header.base_datetime):
                hadm_record = wfdb.io.rdrecord(r, pn_dir=dirname)

                if sig_name in hadm_record.sig_name:
                    break
                else:
                    raise WfDataNotFoundError(
                        f"Record found, but signal {sig_name} not in available record signals: {hadm_record.sig_name}"
                    )
        else:
            raise WfDataNotFoundError(f"No record found for hadm: {hadm_id}")

        return hadm_record

    def precheck_hadm_has_data(self, hadm_id: int, sig_name: str) -> bool:
        """
        Does the minimum amount of work possible to check if hadm has usable data

        Does not download the actual record, so does not check if the data has appropriate variance.
        Only checks if data exists
        """
        try:
            subject_id = self._hadm_to_subject(hadm_id)
            dirname = self._sid_to_dir(subject_id)

            try:
                record_list = wfdb.get_record_list(dirname)
            except ValueError:
                raise WfDataNotFoundError(
                    f"No record found whatsoever for hadm id: {hadm_id}"
                )

            dated_records = [r for r in record_list if "-" in r and "n" not in r]

            for r in dated_records:
                # Here we just look at the layout subheader, instead of downloading the entire record
                header = wfdb.io.rdheader(r, pn_dir=dirname)

                if hadm_id == self._subject_to_hadm(subject_id, header.base_datetime):
                    # TODO: should only ever be one of these, maybe add a check later?
                    layout_header_name = [
                        seg for seg in header.seg_name if "_layout" in seg
                    ][0]

                    layout_header = wfdb.io.rdheader(layout_header_name, pn_dir=dirname)

                    if sig_name in layout_header.sig_name:
                        break
                    else:
                        raise WfDataNotFoundError(
                            f"Record found, but signal {sig_name} not in available record signals: {layout_header.sig_name}"
                        )
            else:
                raise WfDataNotFoundError(f"No record found for hadm: {hadm_id}")

            return True
        except WfDataNotFoundError:
            return False

    def get_wf_anyinterval(
        self, hadm_id: int, sig_name: str, duration: datetime.timedelta
    ) -> np.ndarray:
        hadm_record = self._get_record_by_hadm(hadm_id, sig_name)

        signal_idx = hadm_record.sig_name.index(sig_name)
        signal_raw = hadm_record.p_signal[:, signal_idx]

        # TODO: assuming fs (sampling frequency) is in Hz, must verify
        duration_indices = int(duration.total_seconds()) * hadm_record.fs

        return self._get_valid_signal(signal_raw, duration_indices)


if __name__ == "__main__":
    wff = WfFetcher()
    sample_hadm_id = 182220

    ret = wff.get_wf_anyinterval(sample_hadm_id, "II", datetime.timedelta(minutes=5))

    print("Done")
