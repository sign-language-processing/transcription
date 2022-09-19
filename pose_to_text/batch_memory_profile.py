import argparse
import logging
import torch
from tqdm import tqdm

from joeynmt.helpers import parse_train_args

from pose_to_text.train import setup_training

logger = logging.getLogger(__name__)


def profile(cfg_file: str):
    cfg, model, train_data, *_ = setup_training(cfg_file)

    device = torch.device("cuda")
    model.to(device)
    train_iter = train_data.make_iter(batch_size=1, device=device)

    (  # pylint: disable=unbalanced-tuple-unpacking
        model_dir,
        load_model,
        load_encoder,
        load_decoder,
        loss_type,
        label_smoothing,
        normalization,
        learning_rate_min,
        keep_best_ckpts,
        logging_freq,
        validation_freq,
        log_valid_sents,
        early_stopping_metric,
        seed,
        shuffle,
        epochs,
        max_updates,
        batch_size,
        batch_type,
        batch_multiplier,
        device,
        n_gpu,
        num_workers,
        fp16,
        reset_best_ckpt,
        reset_scheduler,
        reset_optimizer,
        reset_iter_state,
    ) = parse_train_args(cfg["training"])

    model.log_parameters_list()
    model.loss_function = (loss_type, label_smoothing)

    with open("profile/output.txt", "w") as f:

        for batch in train_iter:
            torch.cuda.reset_peak_memory_stats()
            src_len = len(batch.src[0])
            trg_len = len(batch.trg[0])
            model(return_type="loss", **vars(batch))
            total_memory_use = torch.cuda.max_memory_allocated()
            line = f"src_len={src_len}, trg_len={trg_len}, memory={total_memory_use}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    profile(cfg_file=args.config)
