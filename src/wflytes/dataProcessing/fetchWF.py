"""
Download relevant waveforms from physionet
"""
import wfdb
import dask.dataframe as dd

if __name__ == "__main__":
    a = wfdb.rdheader("3544749_0001", pn_dir="mimic3wdb-matched/1.0/p00/p000020")
    print(a.sig_name)
    print(a.sig_len)
    print()
    print("done")
