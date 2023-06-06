import os
from typing import Dict, Any, Tuple, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from _shared.collator import zero_pad_collator

from .args import args
from .data import get_dataset, PoseSegmentsDataset
from .model import PoseTaggingModel


def get_train_dataset(data_args: Dict[str, Any]) -> Tuple[Optional[PoseSegmentsDataset], Optional[DataLoader]]:
    if not args.train:
        return None, None

    split = "validation" if args.data_dev else "train"
    shuffle = False if args.data_dev else True
    dataset = get_dataset(split=split, **data_args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        collate_fn=zero_pad_collator)

    return dataset, loader


def get_validation_dataset(data_args: Dict[str, Any]) -> Tuple[Optional[PoseSegmentsDataset], Optional[DataLoader]]:
    if not args.train and not args.test:
        return None, None

    dataset = get_dataset(split="validation", **data_args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size_devtest,
                        shuffle=False,
                        collate_fn=zero_pad_collator)

    return dataset, loader


def get_test_dataset(data_args: Dict[str, Any]) -> Tuple[Optional[PoseSegmentsDataset], Optional[DataLoader]]:
    if not args.test:
        return None, None

    dataset = get_dataset(split="test", **data_args)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size_devtest,
                        shuffle=False,
                        collate_fn=zero_pad_collator)

    return dataset, loader


def init_model(train_dataset: PoseSegmentsDataset, test_dataset: PoseSegmentsDataset = None):
    any_dataset = train_dataset if train_dataset is not None else test_dataset
    _, num_pose_joints, num_pose_dims = any_dataset[0]["pose"]["data"].shape

    model_args = dict(pose_dims=(num_pose_joints, num_pose_dims),
                      pose_projection_dim=args.pose_projection_dim,  
                      hidden_dim=args.hidden_dim,
                      encoder_depth=args.encoder_depth,
                      encoder_bidirectional=args.encoder_bidirectional,
                      learning_rate=args.learning_rate,
                      lr_scheduler=args.lr_scheduler)

    if args.weighted_loss and train_dataset is not None:
        model_args['sign_class_weights'] = train_dataset.inverse_classes_ratio("sign")
        model_args['sentence_class_weights'] = train_dataset.inverse_classes_ratio("sentence")

    print("Model Arguments:", model_args)

    if args.checkpoint is not None:
        return PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)

    return PoseTaggingModel(**model_args)


if __name__ == '__main__':
    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="pose-to-segments", log_model=False, offline=False)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)

    data_args = dict(poses=args.pose,
                     fps=args.fps,
                     components=args.pose_components,
                     reduce_face=args.pose_reduce_face,
                     hand_normalization=args.hand_normalization,
                     optical_flow=args.optical_flow,
                     only_optical_flow=args.only_optical_flow,
                     classes=args.classes,
                     data_dir=args.data_dir)
    
    train_dataset, train_loader = get_train_dataset(data_args)
    validation_dataset, validation_loader = get_validation_dataset(data_args)
    test_dataset, test_loader = get_test_dataset(data_args)

    model = init_model(train_dataset, test_dataset)

    callbacks = [
        EarlyStopping(monitor='validation_frame_f1_avg', patience=20, verbose=True, mode='max'),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(
            ModelCheckpoint(dirpath=f"models/{LOGGER.experiment.name}",
                            filename='best',
                            verbose=True,
                            save_top_k=1,
                            save_last=True,
                            monitor='validation_frame_f1_avg',
                            every_n_epochs=1,
                            mode='max'))

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=LOGGER,
                         callbacks=callbacks,
                         log_every_n_steps=(1 if args.data_dev else 10),
                         accelerator='gpu',
                         check_val_every_n_epoch=1,
                         devices=args.gpus)

    if args.train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    if args.test:
        if args.train:
            # automatically auto-loads the best weights from the previous run
            # see: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#testing
            trainer.test(dataloaders=validation_loader)
            trainer.test(dataloaders=test_loader)
        else:
            trainer.test(model, dataloaders=validation_loader)
            trainer.test(model, dataloaders=test_loader)


    if args.save_jit:
        # TODO: how to automatically load the best weights like above?
        pose_data = torch.randn((1, 100, *model.pose_dims))
        traced_cell = torch.jit.trace(model, tuple([pose_data]), strict=False)
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../dist", "model.pth")
        torch.jit.save(traced_cell, model_path)
