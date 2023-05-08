# SignDiffusion - A Generic Framework for Sign Language Animation

## Introduction

Building on the success of Ham2Pose @shalev2022ham2pose, which animates HamNoSys notation into pose sequences, we propose a generic framework that supports various conditioning, including HamNoSys, SignWriting, and generic text. The goal is to improve sign language animation by addressing the limitations of the Ham2Pose approach and developing a more generic and flexible framework.

## Objectives

1. Improve the diffusion process used in Ham2Pose.
2. Enhance the training method for scalability.
3. Optimize model evaluation using appropriate metrics.
4. Utilize both parallel and monolingual pose data for training.
5. Implement a better sequence length prediction method.
6. Conduct hyperparameter tuning using WANDB sweeps.

## Proposed Solutions

| Category          | Problem                                                                                                      | Proposed Solution                                                                                                   |
|-------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Diffusion Process | Directly predicting $T_{i-1}$ from step $T_i$, instead of predicting $T_0$ and sampling $T_{i-1}$ from it.   | Following a standard diffusion process: sampling a step $T_i$, noising the pose to that step, and predicting $T_0$. |
| Training Method   | Training by looping over all steps for a single example, not scalable to many diffusion steps.               | Sampling a step for each iteration during training.                                                                 |
| Model Evaluation  | Stopping model training based on the loss, not the proposed metrics.                                         | Calculating the DTW-MJE metric and stopping training when it stops improving.                                       |
| Training Data     | Only using parallel corpora for training, missing monolingual pose data.                                     | Using both parallel and monolingual pose data for training.                                                         |
| Sequence Length   | Predicting sequence length as a number using regression, leading to average predictions.                     | Learning to predict a sequence length distribution (mu, std) and sampling from it.                                  |
| Hyperparameter    | No hyper-parameter search                                                                                    | We perform hyperparameter tuning using WANDB sweeps.                                                                |

## Datasets

We use the following datasets for training our model:

| Dataset         | Citation                           | Notation     | Number of Videos | Video Length (mean ± std) |
|-----------------|------------------------------------|--------------|------------------|---------------------------|
| Sign2MINT       |                                    | SignWriting  | \fix{@@}         | \fix{@@ ± @@}             |
| DictaSign       | @dataset:matthes2012dicta          | HamNoSys     | \fix{@@}         | \fix{@@ ± @@}             |
| DGS_Types       | @dataset:hanke-etal-2020-extending | HamNoSys     | \fix{@@}         | \fix{@@ ± @@}             |
| AUTSL           | @dataset:sincan2020autsl           |              | \fix{@@}         | \fix{@@ ± @@}             |

## Evaluation

### Quantitative Evaluation

We evaluate our model using the DTW-MJE metric, as proposed in @huang2021towards. We assess the model on multiple datasets using the distance metric:

- Sign2MINT (parallel SignWriting and Poses)
- DictaSign (parallel HamNoSys and Poses), comparing with Ham2Pose
- DGS_Types (parallel HamNoSys and Poses), comparing with Ham2Pose

### Qualitative Evaluation

We include a figure with two subfigures:

1. SignWriting example, original Pose Sequence, and predicted Pose Sequence
2. HamNoSys example, original Pose Sequence, and predicted Pose Sequence
