# Generating Data

To train a model using this repo you first have to generate the data. Follow the instruction
in [../data/README.md](../data/README.md).

Now that the data exists in the `raw` directories, it has to be tokenized and split into train, test and devtest data.
To do so run:

```bash
python ../data/__init__.py
```

This will generate a `compressed` directory with the data in the right form.

# Building the image:

- run `nvidia-smi`
    - change "FROM" according to your CUDA version
    - change "gpu=4" and "numgpus=4" according to your GPU count
- build the image:
    - run `nvidia-docker build . -t bergamot`

# Running the image:

First, mark the number of GPUs you have in `./firefox-translations-training/profiles/custom/config.yaml`

Second, create a empty directory with the name `training` in `text_to_text`. This will be mounted to and filled by the
Docker image.

Then, use `nvidia-docker` to run the image,
copying the `profile` and `config` files into the container,
and mounting the data and training directory.

Please keep in mind that if you didn't generate all the data you have to adjust
the `./firefox-translations-training/configs/config.spoken-to-signed.yml` file and remove all the datasets that you
didn't generate under train, devtest and test.

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

Once in the container, you can test it by using the test script (should take a few hours to a day)

```bash
make test
```

To run the pipeline on your own data, specify paths to config files

```bash
# Because of a bug in the mono-corpus code, create a temporary directory
mkdir -p /training/data/spoken-signed/spoken_to_signed/original/mono/custom-mono_/custom_corpus/common_words/original/custom-mono_/custom_corpus/common_words/
make run PROFILE=custom CONFIG=configs/config.spoken-to-signed.yml
```

To reset the training for all experiments, just delete the training directory from within the container

```bash
rm -r /training/*
```