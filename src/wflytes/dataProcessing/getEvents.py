"""
Cut all relevant events out of main chartevents file and save to smaller file
"""
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from wflytes.dataProcessing import potassium_events

chartevent_columns = {
    "SUBJECT_ID": "int",
    "HADM_ID": "int",
    "ICUSTAY_ID": "float",
    "ITEMID": "int",
    "CHARTTIME": "object",
    "STORETIME": "object",
    "VALUENUM": "float",
    "RESULTSTATUS": "object",
    "STOPPED": "object",
    "VALUE": "object",
}

if __name__ == "__main__":
    ce = dd.read_csv(
        "mimiciii/CHARTEVENTS.csv",
        # columns=chartevent_columns.keys(),
        dtype=chartevent_columns,
        assume_missing=True,
    )
    ProgressBar().register()
    ce = ce[ce["ITEMID"].isin(potassium_events)].compute(scheduler="processes")

    print(ce["ITEMID"].value_counts())

    ce[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "ITEMID",
            "CHARTTIME",
            "STORETIME",
            "VALUENUM",
        ]
    ].to_csv("cache/potassiumevents.csv")
