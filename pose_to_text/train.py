import argparse
import logging
import shutil
from pathlib import Path

from joeynmt.helpers import load_config, log_cfg, make_logger, make_model_dir, set_seed
from joeynmt.prediction import test
from joeynmt.training import TrainManager

from pose_to_text.dataset import get_dataset
from pose_to_text.model import build_model

logger = logging.getLogger(__name__)


def train(cfg_file: str, skip_test: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.
    :param cfg_file: path to configuration yaml file
    :param skip_test: whether a test should be run or not after training
    """
    # read config file
    cfg = load_config(Path(cfg_file))

    # make logger
    model_dir = make_model_dir(
        Path(cfg["training"]["model_dir"]),
        overwrite=cfg["training"].get("overwrite", False),
    )
    joeynmt_version = make_logger(model_dir, mode="train")
    if "joeynmt_version" in cfg:
        assert str(joeynmt_version) == str(
            cfg["joeynmt_version"]), (f"You are using JoeyNMT version {joeynmt_version}, "
                                      f'but {cfg["joeynmt_version"]} is expected in the given config.')

    # write all entries of config to the log
    log_cfg(cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    data_args = {
        "poses": cfg["data"]["pose"],
        "fps": cfg["data"]["fps"],
        "components": cfg["data"]["components"],
        "max_seq_size": cfg["data"]["max_seq_size"]
    }
    train_data = get_dataset(**data_args, split="train[50:]")
    dev_data = get_dataset(**data_args, split="train[:50]")
    test_data = dev_data

    trg_vocab = train_data.trg_vocab

    trg_vocab.to_file(model_dir / "trg_vocab.txt")
    if hasattr(train_data.tokenizer[train_data.trg_lang], "copy_cfg_file"):
        train_data.tokenizer[train_data.trg_lang].copy_cfg_file(model_dir)

    # build an encoder-decoder model
    _, num_pose_joints, num_pose_dims = train_data[0][0].shape
    model = build_model(pose_dims=(num_pose_joints, num_pose_dims), cfg=cfg["model"], trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, cfg=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if not skip_test:
        # predict with the best model on validation and test
        # (if test data is available)

        ckpt = model_dir / f"{trainer.stats.best_ckpt_iter}.ckpt"
        output_path = model_dir / f"{trainer.stats.best_ckpt_iter:08d}.hyps"

        datasets_to_test = {
            "dev": dev_data,
            "test": test_data,
            "src_vocab": None,
            "trg_vocab": trg_vocab,
        }
        test(
            cfg_file,
            ckpt=ckpt.as_posix(),
            output_path=output_path.as_posix(),
            datasets=datasets_to_test,
        )
    else:
        logger.info("Skipping test after training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    train(cfg_file=args.config)
