"""
Download a test set of all stroke patients and an equally large cohort of controls
"""
import pandas as pd
from tqdm import tqdm

from wflytes.dataProcessing.wfFetcher import WfDataNotFoundError, WfFetcher


def get_stroke_hdms():
    diagnoses = pd.read_csv("mimiciii/DIAGNOSES_ICD.csv")
    diagnoses = diagnoses.dropna(subset="ICD9_CODE")
    stroke_hadm_ids = diagnoses[diagnoses["ICD9_CODE"].str.startswith("431")][
        "HADM_ID"
    ].unique()

    return stroke_hadm_ids


if __name__ == "__main__":
    wff = WfFetcher()

    # Pull stroke patient hadm_ids
    stroke_hadm_ids = get_stroke_hdms()

    success_count = 0
    for hadm_id in tqdm(stroke_hadm_ids):
        try:
            wff.cache_locally(hadm_id)
            success_count += 1
        except WfDataNotFoundError:
            pass

    print(
        f"[*] Successfully downloaded {success_count} records out of {len(stroke_hadm_ids)}"
    )

    # Pull a subset of the rest of the db that definitely DOES NOT contain stroke patients
    admissions = pd.read_csv("mimiciii/ADMISSIONS.csv")
    non_stroke_hadm_ids = admissions[~admissions["HADM_ID"].isin(stroke_hadm_ids)]

    # Fetch raw data with WF Fetcher

    # Save to directory structure

    print("[+] Done")
