import os
from utils import data_sets_utils
from utils import model_utils
import numpy as np

np.random.seed(133)

pickle_file = os.path.join('../', 'notMNIST.pickle')
pickle_file_sanit = os.path.join('../', 'notMNIST-sanit.pickle')

(train_data_set, train_labels, test_data_set,
 test_labels, valid_data_set, valid_labels) = data_sets_utils.read_data_pickle(pickle_file)

(train_data_set_sanitized, train_labels_sanitized, test_data_set_sanitized,
 test_labels_sanitized, valid_data_set_sanitized,
 valid_labels_sanitized) = data_sets_utils.read_data_pickle(pickle_file_sanit)

print('CONVOLUTION')

for depth in [16]:
    num_hidden = 64
    learning_rate = 0.05
    print("Learning rate = {0}, {1}, {2}".format(learning_rate, num_hidden, depth))
    np.random.seed(133)
    model_utils.train_tensor_flow_convolution(train_data_set, train_labels, test_data_set,
                                              test_labels, valid_data_set,
                                              valid_labels, learning_rate=learning_rate,
                                              num_steps=200000, num_hidden=num_hidden, depth=depth)