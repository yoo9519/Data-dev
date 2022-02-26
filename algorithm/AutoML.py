##############################
## -- import module --
############################## 
from weakref import finalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.executable)
from tqdm.notebook import tqdm

# import sklearn
from sklearn.preprocessing import StandardScaler

# import AutoML-pycaret
from pycaret.datasets import get_data
dataset = get_data('juice')
from pycaret.classification import *
setup_clf = setup(data=dataset, target='Purchase')
from pycaret.utils import check_metric

# IPython display
from IPython.display import Image




########################
## -- Start Method -- ##
########################
# Pycaret Model Test
print(dataset.shape, dataset.head(5))

top3 = compare_models(sort='Accuracy', n_select=3)
tuned_top3 = [tune_model(i) for i in top3]
blender_top3 = blend_models(estimator_list=tuned_top3)

final_model = finalize_model(blender_top3)
pred = predict_model(final_model, data=dataset.iloc[-100:])
print(pred)



########################
## -- Final Record -- ##
########################
print("Predict Evaluation Score is {}".format(check_metric(pred['Purchase'], pred['Label'], metric='Accuracy')))