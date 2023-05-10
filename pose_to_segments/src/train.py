import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from _shared.collator import zero_pad_collator

from .args import args
from .data import get_dataset
from .model import PoseTaggingModel

if __name__ == '__main__':
    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="pose-to-segments", log_model=False, offline=False)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)

    data_args = dict(poses=args.pose,
                     fps=args.fps,
                     components=args.pose_components,
                     hand_normalization=args.hand_normalization,
                     optical_flow=args.optical_flow,
                     data_dir=args.data_dir)

    if not args.test_only:                 
        if args.data_dev:
            train_dataset = get_dataset(split="validation", **data_args)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=zero_pad_collator)
        else:
            train_dataset = get_dataset(split="train", **data_args)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=zero_pad_collator)

        validation_dataset = get_dataset(split="validation", **data_args)
        validation_loader = DataLoader(validation_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=zero_pad_collator)

    test_dataset = get_dataset(split="test", **data_args)
    test_loader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = test_dataset[0]["pose"]["data"].shape

    # Model Arguments
    model_args = dict(pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      encoder_depth=args.encoder_depth,
                      encoder_bidirectional=args.encoder_bidirectional,
                      learning_rate=args.learning_rate)
    if not args.test_only:
        model_args['sign_class_weights'] = train_dataset.inverse_classes_ratio("sign") 
        model_args['sentence_class_weights'] = train_dataset.inverse_classes_ratio("sentence") 

    print("Model Arguments:", model_args)

    if args.checkpoint is not None:
        model = PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = PoseTaggingModel(**model_args)

    callbacks = [EarlyStopping(monitor='validation_loss', patience=100, verbose=True, mode='min')]

    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(
            ModelCheckpoint(dirpath=f"models/{LOGGER.experiment.name}",
                            filename='best',
                            verbose=True,
                            save_top_k=1,
                            save_last=True,
                            monitor='validation_loss',
                            # every_n_train_steps=32,
                            every_n_epochs=1,
                            mode='min'))

    trainer = pl.Trainer(max_epochs=100,
                        logger=LOGGER,
                        callbacks=callbacks,
                        log_every_n_steps=10,
                        accelerator='gpu',
                        #  val_check_interval=32,
                        check_val_every_n_epoch=1,
                        devices=args.gpus)

    if args.test_only:
        trainer.test(model, dataloaders=test_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

        if args.test:
            trainer.test(dataloaders=test_loader)

    if args.save_jit:
        pose_data = torch.randn((1, 100, num_pose_joints, num_pose_dims))
        traced_cell = torch.jit.trace(model, tuple([pose_data]), strict=False)
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../dist", "model.pth")
        torch.jit.save(traced_cell, model_path)
