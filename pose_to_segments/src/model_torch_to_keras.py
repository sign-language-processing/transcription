import numpy as np
import torch
from pytorch2keras import pytorch_to_keras
from torch.autograd import Variable


def load_torch_model():
    return torch.jit.load("dist/model.pth")


def convert_torch_to_keras(model, shape: tuple):
    input_np = np.random.uniform(0, 1, (1, *[1 if v is None else v for v in shape]))
    input_var = Variable(torch.FloatTensor(input_np))

    return pytorch_to_keras(model, input_var, [shape], verbose=True)  # names='short'


if __name__ == "__main__":
    torch_model = load_torch_model()
    input_vector_size = torch_model.pose_projection.weight.shape[1]

    keras_model = convert_torch_to_keras(torch_model, (None, input_vector_size))
