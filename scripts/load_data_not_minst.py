from utils import file_utils
from utils import plot_utils
from utils import data_sets_utils
from utils import model_utils
import os

url = 'http://commondatastorage.googleapis.com/books1000/'
file_names = [{'name': 'notMNIST_large.tar.gz', 'expected_bytes': 247336696},
              {'name': 'notMNIST_small.tar.gz', 'expected_bytes': 8458043}]

test_file_name = file_utils.maybe_download('notMNIST_small.tar.gz', 8458043,
                                           url=url, data_root='../', force=False)
train_file_name = file_utils.maybe_download('notMNIST_large.tar.gz', 247336696,
                                            url=url, data_root='../', force=False)
train_folders = file_utils.maybe_extract(train_file_name, data_root='../')
test_folders = file_utils.maybe_extract(test_file_name, data_root='../')
plot_utils.plot_samples(train_folders, 10, 'Train Data')
plot_utils.plot_samples(test_folders, 10, 'Test Data')
train_data_sets = file_utils.maybe_pickle(train_folders, 45000, pixel_depth=255, image_size=28)
test_data_sets = file_utils.maybe_pickle(test_folders, 1800, pixel_depth=255, image_size=28)

test_labels = plot_utils.generate_fake_label(plot_utils.load_and_display_pickle(test_data_sets, 10, 'Test Datasets'))
train_labels = plot_utils.generate_fake_label(plot_utils.load_and_display_pickle(train_data_sets, 10, 'Train Datasets'))
plot_utils.plot_balance(train_labels, test_labels)
train_size = 200000
valid_size = 10000
test_size = 10000

valid_data_set, valid_labels, train_data_set, train_labels = data_sets_utils.merge_data_sets(
    train_data_sets, train_size, valid_size, image_size=28)
_, _, test_data_set, test_labels = data_sets_utils.merge_data_sets(test_data_sets, test_size, image_size=28)

print('Training:', train_data_set.shape, train_labels.shape)
print('Validation:', valid_data_set.shape, valid_labels.shape)
print('Testing:', test_data_set.shape, test_labels.shape)

train_data_set, train_labels = data_sets_utils.randomize(train_data_set, train_labels)
test_data_set, test_labels = data_sets_utils.randomize(test_data_set, test_labels)
valid_data_set, valid_labels = data_sets_utils.randomize(valid_data_set, valid_labels)

pickle_file = os.path.join('../', 'notMNIST.pickle')

overlap_test_train = data_sets_utils.extract_overlap_hash_where(test_data_set, train_data_set)
print('Number of overlaps:', len(overlap_test_train.keys()))
plot_utils.display_overlap(overlap_test_train, test_data_set, train_data_set)

test_data_set_sanit, test_labels_sanit = data_sets_utils.sanitize(test_data_set, train_data_set, test_labels)
print('Overlapping images removed from test_dataset: ', len(test_data_set) - len(test_data_set_sanit))
valid_data_set_sanit, valid_labels_sanit = data_sets_utils.sanitize(valid_data_set, train_data_set, valid_labels)
print('Overlapping images removed from valid_dataset: ', len(valid_data_set) - len(valid_data_set_sanit))
print('Training:', train_data_set.shape, train_labels.shape)
print('Validation:', valid_labels_sanit.shape, valid_labels_sanit.shape)
print('Testing:', test_data_set_sanit.shape, test_labels_sanit.shape)
pickle_file_sanit = os.path.join('../', 'notMNIST-sanit.pickle')
data_sets_utils.save_data_pickle(pickle_file, train_data_set, train_labels,
                                 valid_data_set, valid_labels, test_data_set, test_labels)
data_sets_utils.save_data_pickle(pickle_file_sanit, train_data_set, train_labels,
                                 valid_data_set_sanit, valid_labels_sanit, test_data_set_sanit, test_labels_sanit)
for sample_size in [10, 100, 5000]:
    model_utils.train_and_predict(sample_size, train_data_set, train_labels, test_data_set, test_labels, image_size=28)