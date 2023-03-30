import os

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

    train_dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=zero_pad_collator)

    validation_dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="validation")
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    # Model Arguments
    sign_class_weights = train_dataset.inverse_classes_ratio("sign")
    sentence_class_weights = train_dataset.inverse_classes_ratio("sentence")
    model_args = dict(sign_class_weights=sign_class_weights,
                      sentence_class_weights=sentence_class_weights,
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      encoder_depth=args.encoder_depth,
                      encoder_bidirectional=args.encoder_bidirectional,
                      learning_rate=args.learning_rate)

    print("Model Arguments:", model_args)

    if args.checkpoint is not None:
        model = PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = PoseTaggingModel(**model_args)

    callbacks = [EarlyStopping(monitor='validation_loss', patience=100, verbose=True, mode='min')]

    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(
            ModelCheckpoint(dirpath=f"models/{LOGGER.experiment.id}",
                            filename="model",
                            verbose=True,
                            save_top_k=1,
                            monitor='validation_loss',
                            every_n_train_steps=32,
                            mode='min'))

    trainer = pl.Trainer(max_epochs=100,
                         logger=LOGGER,
                         callbacks=callbacks,
                         log_every_n_steps=10,
                         accelerator='gpu',
                         val_check_interval=32,
                         devices=args.gpus)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
