import argparse

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import config_de as config
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.models.classification.wideresnet.std import wideresnet28x10
from torch_uncertainty.models.wrappers.deep_ensembles import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.transforms import RepeatTarget

torch.set_float32_matmul_precision("medium")


class AddAccAlias(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if "val/cls/Acc" in trainer.callback_metrics:
            trainer.callback_metrics["val_cls_Acc"] = trainer.callback_metrics["val/cls/Acc"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--no_fast_dev_run", action="store_true")
    parser.add_argument("--num_estimators", type=int, default=config.NUM_ESTIMATORS)
    parser.add_argument("--max_epochs", type=int, default=config.MAX_EPOCHS)
    return parser.parse_args()


def resolve_fast_dev_run(args):
    if args.no_fast_dev_run:
        return False
    if args.fast_dev_run:
        return True
    return config.FAST_DEV_RUN


def make_datamodule(args):
    return CIFAR100DataModule(
        root=config.DATA_DIR,
        batch_size=args.batch_size,
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
        val_split=config.VAL_SPLIT,
        auto_augment=config.AUTO_AUGMENT,
        num_workers=getattr(config, "NUM_WORKERS", 4),
    )


def make_model(args):
    backbone = wideresnet28x10(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        style=config.STYLE,
        dropout_rate=config.DROPOUT_RATE,
    )

    return deep_ensembles(
        backbone,
        num_estimators=args.num_estimators,
        task="classification",
        reset_model_parameters=True,
    )


def make_optim_recipe(model):
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

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }


def main():
    args = parse_args()
    fast_dev_run = resolve_fast_dev_run(args)

    pl.seed_everything(args.seed)

    datamodule = make_datamodule(args)
    model = make_model(args)
    optim_recipe = make_optim_recipe(model)

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
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices="auto",
        logger=logger,
        fast_dev_run=fast_dev_run,
        max_epochs=args.max_epochs,
        precision=config.PRECISION,
        callbacks=[AddAccAlias(), checkpoint_cb],
    )

    trainer.fit(classifier, datamodule)

    ckpt_path = checkpoint_cb.best_model_path
    print(ckpt_path)
    trainer.test(classifier, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()