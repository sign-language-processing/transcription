# Instructions:
#
# 0. Note your local machine's nvidia cuda version with `nvidia-smi`, and change this file "FROM" accordingly.
#
# 1. Use nvidia-docker (not docker) to build the image
# nvidia-docker build . -t bergamot
#
# 2. Run the image using nvidia-docker, copying your `profile` and `config` files into the container, and mounting your data directory.
# Make sure to mark the number of GPUs you have, and workspace memory to be allocated
# mkdir -p training/experiments/spoken-signed
# mkdir -p training/models/spoken-signed
# nvidia-docker run -it  \
#	--mount type=bind,source="$(pwd)/bergamot/config.spoken-to-signed.yml",target=/firefox-translations-training/configs/config.spoken-to-signed.yml \
#	--mount type=bind,source="$(pwd)/bergamot/profile.yml",target=/firefox-translations-training/profiles/custom/config.yaml \
#	--mount type=bind,source="$(pwd)/data/compressed",target=/custom_corpus \
#	--mount type=bind,source="$(pwd)/training",target=/training \
#	bergamot
#
# 3. To test this works, in your container, use the test script (should take a few hours to a day)
# make test
#
# 4. To run the pipeline on your own data, specify paths to config files
# make run PROFILE=custom CONFIG=configs/config.spoken-to-signed.yml

FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

RUN apt-get update; apt-get update
RUN apt-get install git -y

RUN git clone https://github.com/mozilla/firefox-translations-training.git
WORKDIR firefox-translations-training

RUN chmod +x ./pipeline/setup/install-deps.sh
RUN ./pipeline/setup/install-deps.sh

RUN make conda
RUN make snakemake
RUN make git-modules
RUN make dry-run

# Marian/NCCL is expecting nvcc in specific location
RUN if [ ! -f "/usr/local/cuda/bin/nvcc" ]; then  mkdir /usr/local/cuda/bin/ && ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc; fi
# NCCL: unsupported GNU version! gcc versions later than 6 are not supported!
RUN apt-get install gcc-6 g++-6 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6

# For building purposes, we assume the user has at least 1 gpu, not 8
RUN sed -i 's/^resources:\ gpu=8/resources:\ gpu=1/' profiles/local/config.yaml
RUN sed -i 's/numgpus=8/numgpus=1/' profiles/local/config.yaml
RUN sed -i 's/workspace=12000/workspace=4000/' profiles/local/config.yaml

# Downloads and compiles additional packages
# This command fails do to the gpu not being attached to the container during build time
# However, it is still good to install the dependencies in advanced.
RUN make test; exit 0
