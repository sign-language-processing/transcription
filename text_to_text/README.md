# 📝 ⇝ 🧏 Text-to-Text

Translation between spoken and signed language texts.

## ⚠️ Warning!

- We managed to train this for ASL fingerspelling, but not multilingual fingerspelling.
- This module will change drastically as we probably move to using bergamot.

## Main Idea

We fine tune a machine translation model on data for spoken-to-signed language text translation.

## Extra details

- We use Huggingface transformers for pretrained translation models
- The `run_translation.py` file is minorly adapted
  from [this one](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py)

## setup environment:

```bash
conda create --name hf --no-default-packages python=3.8
conda activate hf
# install torch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# install rest of dependencies
pip install -r requirements.txt
```

## how to run:

1. run `bash text_to_text/scripts/prepare_model.sh` to prepare model
1. run `bash text_to_text/scripts/prepare_data.sh` to prepare data
1. run `bash text_to_text/scripts/train.sh` to train model
1. run `bash text_to_text/scripts/push_to_hub.sh` to push model to the hub