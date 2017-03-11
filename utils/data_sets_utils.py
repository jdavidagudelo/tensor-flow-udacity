import numpy as np
import pickle
import hashlib
import os


def make_arrays(nb_rows, img_size):
    if nb_rows:
        data_set = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        data_set, labels = None, None
    return data_set, labels


def merge_data_sets(pickle_files, train_size, valid_size=0, image_size=0):
    num_classes = len(pickle_files)
    valid_data_set, valid_labels = make_arrays(valid_size, image_size)
    train_data_set, train_labels = make_arrays(train_size, image_size)
    v_size_per_class = valid_size // num_classes
    t_size_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = v_size_per_class, t_size_per_class
    end_l = v_size_per_class + t_size_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_data_set is not None:
                    valid_letter = letter_set[:v_size_per_class, :, :]
                    valid_data_set[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += v_size_per_class
                    end_v += v_size_per_class

                train_letter = letter_set[v_size_per_class:end_l, :, :]
                train_data_set[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += t_size_per_class
                end_t += t_size_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_data_set, valid_labels, train_data_set, train_labels


def randomize(data_set, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_data_set = data_set[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_data_set, shuffled_labels


def extract_overlap_hash_where(data_set_1, data_set_2):
    data_set_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in data_set_1])
    data_set_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in data_set_2])
    overlap = {}
    for i, hash1 in enumerate(data_set_hash_1):
        duplicates = np.where(data_set_hash_2 == hash1)
        if len(duplicates[0]):
            overlap[i] = duplicates[0]
    return overlap


def sanitize(data_set_1, data_set_2, labels_1):
    data_set_hash_1 = np.array([hashlib.sha256(img).hexdigest() for img in data_set_1])
    data_set_hash_2 = np.array([hashlib.sha256(img).hexdigest() for img in data_set_2])
    overlap = []  # list of indexes
    for i, hash1 in enumerate(data_set_hash_1):
        duplicates = np.where(data_set_hash_2 == hash1)
        if len(duplicates[0]):
            overlap.append(i)
    return np.delete(data_set_1, overlap, 0), np.delete(labels_1, overlap, None)


def save_data_pickle(pickle_file, train_data_set, train_labels,
                     valid_data_set, valid_labels, test_data_set, test_labels, force=False):
    if force or not os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'wb') as f:
                save = {
                    'train_data_set': train_data_set,
                    'train_labels': train_labels,
                    'valid_data_set': valid_data_set,
                    'valid_labels': valid_labels,
                    'test_data_set': test_data_set,
                    'test_labels': test_labels,
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise


def read_data_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        train_data_set = data.get('train_data_set')
        train_labels = data.get('train_labels')
        test_data_set = data.get('test_data_set')
        test_labels = data.get('test_labels')
        valid_data_set = data.get('valid_data_set')
        valid_labels = data.get('valid_labels')
    return (train_data_set, train_labels, test_data_set,
            test_labels, valid_data_set, valid_labels)


def reformat(data_set, labels, image_size, num_labels):
    data_set = data_set.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return data_set, labels
