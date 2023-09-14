import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from nurphy.model.transformer import PositionalEncoding
from nurphy.data_utils.utils import subsequent_mask

def visualize_subsequent_mask():
  plt.figure(figsize=(5, 5))
  plt.imshow(subsequent_mask(20)[0])

def visualize_positional_encoding():
  plt.figure(figsize=(15, 5))
  pe = PositionalEncoding(5000, 20, 0)
  x = Variable(torch.zeros(1, 100, 20))

  y = pe.forward(x)
  plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
  plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
  plt.show()