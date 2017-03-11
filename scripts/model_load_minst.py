import os
from utils import data_sets_utils
from utils import model_utils
import numpy as np

np.random.seed(133)

pickle_file = os.path.join('../', 'notMNIST.pickle')
pickle_file_sanit = os.path.join('../', 'notMNIST-sanit.pickle')

(train_data_set, train_labels, test_data_set,
 test_labels, valid_data_set, valid_labels) = data_sets_utils.read_data_pickle(pickle_file)

model_utils.train_and_predict(20000, train_data_set, train_labels, test_data_set, test_labels, image_size=28)

(train_data_set, train_labels, test_data_set,
 test_labels, _, _) = data_sets_utils.read_data_pickle(pickle_file_sanit)

model_utils.train_and_predict(20000, train_data_set, train_labels, test_data_set, test_labels, image_size=28)
