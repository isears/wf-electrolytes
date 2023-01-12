"""
Run Skorch implementation of TST w/specific hyperparams
"""

import glob
import os

import skorch
import torch
import torch.utils.data
from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor
from mvtst.optimizers import AdamW
from sklearn.metrics import average_precision_score, roc_auc_score
from skorch import NeuralNet, NeuralNetBinaryClassifier
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
)

from wflytes.dataProcessing.localWfDataset import LocalWfDataset


# Hack to workaround discrepancies between torch and sklearn shape expectations
# https://github.com/skorch-dev/skorch/issues/442
def my_auroc(net, X, y):
    y_proba = net.predict_proba(X)
    return roc_auc_score(y, y_proba[:, 1])


def my_auprc(net, X, y):
    y_proba = net.predict_proba(X)
    return average_precision_score(y, y_proba[:, 1])


def skorch_tst_factory(params, ds: torch.utils.data.Dataset, pruner=None):
    """
    Generate TSTs wrapped in standard skorch wrapper
    """
    os.makedirs("cache/models/skorchTst", exist_ok=True)

    tst_callbacks = [
        GradientNormClipping(gradient_clip_value=4.0),
        EarlyStopping(patience=3),
        Checkpoint(
            load_best=True,
            fn_prefix="cache/models/skorchTst",
            f_pickle="whole_model.pkl",
        ),
        EpochScoring(my_auroc, name="auroc", lower_is_better=False),
        EpochScoring(my_auprc, name="auprc", lower_is_better=False),
    ]

    if pruner is not None:
        tst_callbacks.append(pruner)

    tst = NeuralNetBinaryClassifier(
        TSTransformerEncoderClassiregressor,
        criterion=torch.nn.BCEWithLogitsLoss,
        iterator_train__collate_fn=ds.collate_skorch,
        iterator_valid__collate_fn=ds.collate_skorch,
        iterator_train__num_workers=1,
        iterator_valid__num_workers=1,
        iterator_train__pin_memory=True,
        iterator_valid__pin_memory=True,
        device="cuda",
        callbacks=tst_callbacks,
        train_split=skorch.dataset.ValidSplit(0.1),
        # TST params
        module__feat_dim=ds.get_num_features(),
        # module__max_len=ds.seq_len,
        module__max_len=490,  # TODO: remove
        max_epochs=25,
        **params,
    )

    return tst


if __name__ == "__main__":
    hadm_ids = [int(h.split("/")[1]) for h in glob.glob("data/*")]
    ds = LocalWfDataset(["II"], all_signals_required=True, hadm_ids=hadm_ids)

    tst_params = {
        "module__d_model": 128,
        "module__n_heads": 16,
        "module__num_layers": 3,
        "module__dim_feedforward": 256,
        "module__num_classes": 1,
        "module__dropout": 0.1,
        "module__pos_encoding": "fixed",
        "module__activation": "gelu",
        "module__norm": "BatchNorm",
        "module__freeze": False,
        "optimizer": AdamW,
        "optimizer__lr": 0.0001,
        "optimizer__weight_decay": 0,
        "iterator_train__batch_size": 8,
        "iterator_valid__batch_size": 8,
    }

    tst = skorch_tst_factory(tst_params, ds)

    tst.fit(ds, y=None)
