"""
Experiment with tsai version of TST
"""
import glob

from sklearn.model_selection import train_test_split
from tsai.all import (
    TST,
    BCEWithLogitsLossFlat,
    Learner,
    MiniRocket,
    RocAucMulti,
    TSDataLoaders,
    computer_setup,
)

from wflytes import config
from wflytes.dataProcessing.localWfDataset import TsaiDataset

if __name__ == "__main__":
    computer_setup()

    hadm_ids = [int(h.split("/")[1]) for h in glob.glob("data/*")]

    train_ids, valid_ids = train_test_split(
        hadm_ids, test_size=0.1, shuffle=True, random_state=42
    )

    # TODO: don't remove everything
    train_ds = TsaiDataset(
        signal_names=["II", "V", "PLETH"], all_signals_required=True, hadm_ids=train_ids
    )
    valid_ds = TsaiDataset(
        signal_names=["II", "V", "PLETH"], all_signals_required=True, hadm_ids=valid_ids
    )

    dl = TSDataLoaders.from_dsets(
        train_ds, valid_ds, bs=32, num_workers=config.cores_available
    )

    model = MiniRocket(len(train_ds.signal_names), 1, 1000 - 20)
    learn = Learner(
        dl, model, loss_func=BCEWithLogitsLossFlat(), metrics=[RocAucMulti()]
    )
    suggested_lr = learn.lr_find().valley

    learn.fit_one_cycle(5, lr_max=suggested_lr)
