import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.executable, '\n', sys.argv[0])

from pycaret.datasets import get_data
from pycaret.clustering import *
from pycaret.utils import check_metric
# from pycaret.utils import enable_colab

from IPython.display import Image


df = get_data('jewellery')
print(df.head(5))

setup_clf = setup(data=df, normalize=True)

# top3_models = compare_models(n_select=3)
#
# model = [create_model(_) for _ in top3_models]
# print(model)

kmeans = create_model('kmeans')
print(kmeans)

# evaluate_model(kmeans)
plot_model(kmeans,plot='elbow')