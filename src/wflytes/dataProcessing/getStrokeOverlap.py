"""
Determine which hemorrhagic stroke patients have overlap
"""
import pandas as pd
from tqdm import tqdm

from wflytes.dataProcessing.wfFetcher import WfFetcher

if __name__ == "__main__":
    diagnoses = pd.read_csv("mimiciii/DIAGNOSES_ICD.csv")
    diagnoses = diagnoses.dropna(subset="ICD9_CODE")
    hadm_ids = diagnoses[diagnoses["ICD9_CODE"].str.startswith("431")][
        "HADM_ID"
    ].unique()

    print(f"[+] Got {len(hadm_ids)} examples")

    wff = WfFetcher()
    usable_hadm_ids = list()

    for hadm_id in tqdm(hadm_ids):
        if wff.precheck_hadm_has_data(hadm_id, "II"):
            usable_hadm_ids.append(hadm_id)

    print(usable_hadm_ids)

    print("Done")
