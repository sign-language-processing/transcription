import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from shared.collator import zero_pad_collator

from .args import args
from .data import get_dataset
from .model import PoseTaggingModel

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="pose-to-segments", log_model=False, offline=False)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)

    train_dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="train[10:]")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=zero_pad_collator)

    validation_dataset = get_dataset(poses=args.pose, fps=args.fps, components=args.pose_components, split="train[:10]")
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    # Model Arguments
    model_args = dict(pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args.hidden_dim,
                      encoder_depth=args.encoder_depth)

    if args.checkpoint is not None:
        model = PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = PoseTaggingModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(
            ModelCheckpoint(dirpath="models/" + LOGGER.experiment.id,
                            filename="model",
                            verbose=True,
                            save_top_k=1,
                            monitor='validation_loss',
                            mode='min'))

    trainer = pl.Trainer(max_epochs=100, logger=LOGGER, callbacks=callbacks, log_every_n_steps=10, gpus=args.gpus)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
