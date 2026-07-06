import argparse

import config_de as config
import lightning as pl
import torch
from torch import nn, optim

from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.models.classification.wideresnet.std import wideresnet28x10
from torch_uncertainty.models.wrappers.deep_ensembles import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

torch.set_float32_matmul_precision("medium")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                   help=f"Batch size (default: config.BATCH_SIZE={config.BATCH_SIZE}).")
    p.add_argument("--fast_dev_run", action="store_true",
                   help=f"If set, enable fast_dev_run (default: config.FAST_DEV_RUN={config.FAST_DEV_RUN}).")
    p.add_argument("--no_fast_dev_run", action="store_true",
                   help="If set, force fast_dev_run=False (overrides config and --fast_dev_run).")
    p.add_argument("--num_estimators", type=int, default=config.NUM_ESTIMATORS,
                   help=f"Number of estimators (default: config.NUM_ESTIMATORS={config.NUM_ESTIMATORS}).")
    p.add_argument("--max_epochs", type=int, default=config.MAX_EPOCHS,
                   help=f"Max epochs (default: config.MAX_EPOCHS={config.MAX_EPOCHS}).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # fast_dev_run logic:
    # - if --fast_dev_run => True
    # - if --no_fast_dev_run => False (priority)
    # - else => config.FAST_DEV_RUN
    fast_dev_run = config.FAST_DEV_RUN
    if args.fast_dev_run:
        fast_dev_run = True
    if args.no_fast_dev_run:
        fast_dev_run = False
        
    pl.seed_everything(args.seed)

    dm = CIFAR100DataModule(
        root=config.DATA_DIR,
        batch_size=args.batch_size,
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
        val_split=config.VAL_SPLIT,
        auto_augment=config.AUTO_AUGMENT,
        num_workers=getattr(config, "NUM_WORKERS", 4),
    )

    model_nn = wideresnet28x10(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        style=config.STYLE,
        dropout_rate=config.DROPOUT_RATE,
    )

    model = deep_ensembles(
        model_nn,
        num_estimators=args.num_estimators,
        task="classification",
        reset_model_parameters=True,
    )
    from lightning.pytorch.callbacks import Callback

    class AddAccAlias(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if "val/cls/Acc" in trainer.callback_metrics:
                trainer.callback_metrics["val_cls_Acc"] = trainer.callback_metrics["val/cls/Acc"]

    checkpoint_cb = ModelCheckpoint(
        monitor="val/cls/Acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
        filename="best-acc-{epoch:03d}-{val_cls_Acc:.4f}",
    )
    
    logger = TensorBoardLogger(
        save_dir="logs",
        name="wideresnet28x10",
        version=None,
        #version=f"seed_{args.seed}",
        default_hp_metric=False,
    )
    
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices="auto",
        logger=logger,
        fast_dev_run=fast_dev_run,
        max_epochs=args.max_epochs,
        precision=config.PRECISION,
        callbacks=[AddAccAlias(),checkpoint_cb],
    )
    

    optimizer = SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=config.NESTEROV,
    )
    
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.MILESTONES,
        gamma=config.GAMMA,
    )

    optim_recipe = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }

    classifier = ClassificationRoutine(
        model=model,
        num_classes=config.NUM_CLASSES,
        loss=nn.CrossEntropyLoss(),
        is_ensemble=True,
        optim_recipe=optim_recipe,
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
        format_batch_fn=RepeatTarget(args.num_estimators),
    )

    trainer.fit(classifier, dm)

    # test best checkpoint (if any)
    ckpt_path = checkpoint_cb.best_model_path
    print(ckpt_path)
    trainer.test(classifier, datamodule=dm, ckpt_path=ckpt_path)
