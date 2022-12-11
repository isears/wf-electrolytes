"""
Determine which lytes events have corresponding Signals
"""
import datetime

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

if __name__ == "__main__":

    events = pd.read_parquet("cache/potassiumevents.parquet")
    sigtimes = pd.read_parquet("cache/sigtimes.parquet")
    sigtimes = sigtimes.reset_index(drop=True)
    ProgressBar().register()

    # TODO: debug only
    # events = events.head(10000)
    # sigtimes = sigtimes.head(10000)

    # Force both dfs to have the same format
    sigtimes = sigtimes.dropna(axis=0)
    events = events.dropna(axis=0)
    sigtimes["VALUENUM"] = pd.NA
    events["starttime"] = events["CHARTTIME"]
    events["endtime"] = events["CHARTTIME"]
    # TODO: when analyzing more events can take this out
    events["sig_name"] = "potassium"
    events = events.drop(
        columns=[c for c in events.columns if c not in sigtimes.columns]
    )
    # Need to know the names of the actual waveforms for later
    waveform_names = sigtimes["sig_name"].unique()

    sig_events_df = pd.concat([sigtimes, events])
    print("[*] Initial single-core preprocessing complete")

    sig_events_df = dd.from_pandas(sig_events_df, chunksize=10000)

    def grouper(group):
        max_distance = datetime.timedelta(hours=12)

        signals = group[group["sig_name"].isin(waveform_names)]
        events = group[~group["sig_name"].isin(waveform_names)]

        def matcher(signal_row):
            relevent_evnets = events[
                (((signal_row["starttime"] - events["starttime"]) < max_distance))
                & ((events["starttime"] - (signal_row["endtime"]) < max_distance))
            ]

            return (
                relevent_evnets["VALUENUM"].to_list(),
                relevent_evnets["starttime"].to_list(),
            )

        if len(signals) > 0 and len(events) > 0:
            signals[["VALUENUM", "eventtime"]] = signals.apply(
                matcher, axis=1, result_type="expand"
            )
            signals = signals.explode(["VALUENUM", "eventtime"])
        else:
            return pd.DataFrame(
                columns=[
                    "SUBJECT_ID",
                    "starttime",
                    "endtime",
                    "sig_name",
                    "VALUENUM",
                    "eventtime",
                ]
            )

        return signals

    res = (
        sig_events_df.groupby("SUBJECT_ID")
        .apply(
            grouper,
            meta=pd.DataFrame(
                columns=[
                    "SUBJECT_ID",
                    "starttime",
                    "endtime",
                    "sig_name",
                    "VALUENUM",
                    "eventtime",
                ]
            ),
        )
        .compute(scheduler="processes")
    )

    # For some reason this only happens under a debugger :/
    # res = res.droplevel(level=0)

    res = res.dropna(subset=["VALUENUM"])
    # Save non-unique index that uniquely identifies each signal
    # There may be multiple copies of a signal if there are multiple contemperaneous measurements
    # The number of duplications of a signal correspond to the number of contemperaneous measurements
    res["sig_idx"] = res.index

    print(res.head())
    res.to_parquet("cache/enriched.parquet", index=False)
    print("Done")
