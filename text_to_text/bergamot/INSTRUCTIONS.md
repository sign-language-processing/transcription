# Building the image:

- run `nvidia-smi` 
  - change "FROM" according to your CUDA version
  - change "gpu=4" and "numgpus=4" according to your GPU count
- build the image:
  - run `nvidia-docker build . -t bergamot`

# Running the image:

first, mark the number of GPUs you have in `firefox-translations-training/profiles/custom/config.yaml`

then, use `nvidia-docker` to run the image, 
copying the `profile` and `config` files into the container, 
and mounting the data directory.
```bash
nvidia-docker run -it  \
	--mount type=bind,source="$(pwd)/bergamot/firefox-translations-training/configs/config.spoken-to-signed.yml",target=/firefox-translations-training/configs/config.spoken-to-signed.yml \
	--mount type=bind,source="$(pwd)/bergamot/firefox-translations-training/profiles/custom/config.yaml",target=/firefox-translations-training/profiles/custom/config.yaml \
	--mount type=bind,source="$(pwd)/bergamot/firefox-translations-training/pipeline/clean/clean-corpus.sh",target=/firefox-translations-training/pipeline/clean/clean-corpus.sh \
	--mount type=bind,source="$(pwd)/bergamot/firefox-translations-training/pipeline/clean/clean-mono.sh",target=/firefox-translations-training/pipeline/clean/clean-mono.sh \
	--mount type=bind,source="$(pwd)/bergamot/firefox-translations-training/pipeline/train/spm-vocab.sh",target=/firefox-translations-training/pipeline/train/spm-vocab.sh \
	--mount type=bind,source="$(pwd)/data/compressed",target=/custom_corpus \
	--mount type=bind,source="$(pwd)/training",target=/training \
	bergamot
```

to test this works, in your container, use the test script (should take a few hours to a day)
```bash
make test
```

to run the pipeline on your own data, specify paths to config files
```bash
# Because of a bug in the mono-corpus code, create a temporary directory
mkdir -p /training/data/spoken-signed/spoken_to_signed/original/mono/custom-mono_/custom_corpus/common_words/original/custom-mono_/custom_corpus/common_words/
make run PROFILE=custom CONFIG=configs/config.spoken-to-signed.yml
```

to reset the training for all experiments, just delete the training directory from within the container
```bash
rm -r /training/*
```