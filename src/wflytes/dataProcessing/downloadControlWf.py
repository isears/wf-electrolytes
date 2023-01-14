"""
Download a test set of all stroke patients and an equally large cohort of controls
"""
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from wflytes.dataProcessing.downloadStrokeWf import get_stroke_hdms
from wflytes.dataProcessing.wfFetcher import WfDataNotFoundError, WfFetcher

if __name__ == "__main__":
    wff = WfFetcher()

    stroke_hadm_ids = get_stroke_hdms()
    admissions = pd.read_csv("mimiciii/ADMISSIONS.csv")
    non_stroke_hadm_ids = admissions[~admissions["HADM_ID"].isin(stroke_hadm_ids)]
    non_stroke_hadm_ids["has_data"] = non_stroke_hadm_ids["HADM_ID"].progress_apply(
        wff.precheck_hadm_has_data
    )
    non_stroke_hadm_ids = non_stroke_hadm_ids[non_stroke_hadm_ids["has_data"]]
    non_stroke_hadm_ids = non_stroke_hadm_ids.sample(1000, random_state=42)

    success_count = 0
    for hadm_id in tqdm(non_stroke_hadm_ids["HADM_ID"].unique()):
        try:
            wff.cache_locally(hadm_id)
            success_count += 1
        except WfDataNotFoundError:
            pass

    print(
        f"[*] Successfully downloaded {success_count} records out of {len(stroke_hadm_ids)}"
    )

    print("[+] Done")
