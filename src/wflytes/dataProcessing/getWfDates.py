"""
Gather waveform start / end dates and compile into searchable database
"""
import datetime

import pandas as pd
import wfdb
from tqdm import tqdm

if __name__ == "__main__":
    dirs = wfdb.get_record_list("mimic3wdb-matched")
    db = pd.DataFrame()

    for d in tqdm(dirs):
        icustays = [
            r
            for r in wfdb.get_record_list(f"mimic3wdb-matched/1.0/{d}")
            if "-" in r and "n" not in r
        ]

        for stay_record in icustays:
            # Alternatively, rd_segments=True to automatically get all segment headers
            records = wfdb.io.rdheader(stay_record, pn_dir=f"mimic3wdb-matched/1.0/{d}")

            # TODO: may not even need this...
            layout_name = [s for s in records.seg_name if "_layout" in s]
            assert len(layout_name) == 1  # Should only ever be 1 layout file
            layout_name = layout_name[0]
            layout_rec = wfdb.io.rdheader(
                layout_name, pn_dir=f"mimic3wdb-matched/1.0/{d}"
            )

            actual_records = [
                r for r in records.seg_name if "_layout" not in r and r != "~"
            ]

            for rec_name in actual_records:
                try:
                    rec_header = wfdb.io.rdheader(
                        rec_name, pn_dir=f"mimic3wdb-matched/1.0/{d}"
                    )

                    starttime = datetime.datetime.combine(
                        records.base_date, rec_header.base_time
                    )

                    # TODO: I don't think this is right
                    endtime = starttime + datetime.timedelta(
                        seconds=(rec_header.sig_len / rec_header.fs)
                    )

                    compiled_df = pd.DataFrame(
                        data={
                            "SUBJECT_ID": [int(d.split("p")[-1][:-1])]
                            * rec_header.n_sig,
                            "starttime": [starttime] * rec_header.n_sig,
                            "endtime": [endtime] * rec_header.n_sig,
                            "sig_name": rec_header.sig_name,
                        }
                    )

                    db = pd.concat([db, compiled_df])
                except wfdb.io._url.NetFileNotFoundError as e:
                    print(f"Couldn't locate {rec_name}: {e}")

    db.to_parquet("cache/sigtimes.parquet")
