import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.8, random_state=101)

print(data)
