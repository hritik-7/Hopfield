import numpy as np

from torchvision import transforms
from scipy.special import softmax
from scipy.special import logsumexp


print("SOFTY",softmax(np.array([1, 2, 3])))
print("SOFTY",softmax(np.array([1, 2, 3])).sum())