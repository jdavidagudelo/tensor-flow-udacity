import os
import matplotlib.pyplot as plt
import random
from matplotlib import image
import numpy as np
import pickle


def plot_samples(data_folders, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    for folder in data_folders:
        image_files = os.listdir(folder)
        image_sample = random.sample(image_files, sample_size)
        for i in image_sample:
            image_file = os.path.join(folder, i)
            ax = fig.add_subplot(len(data_folders), sample_size, sample_size * data_folders.index(folder) +
                                 image_sample.index(i) + 1)
            i = image.imread(image_file)
            ax.imshow(i)
            ax.set_axis_off()

    plt.show()


def generate_fake_label(sizes):
    labels = np.ndarray(sum(sizes), dtype=np.int32)
    end = 0
    for label, size in enumerate(sizes):
        start = end
        end += size
        for j in range(start, end):
            labels[j] = label
    return labels


def plot_balance(train_labels, test_labels):
    fig, ax = plt.subplots(1, 2)
    bins = np.arange(train_labels.min(), train_labels.max() + 2)
    ax[0].hist(train_labels, bins=bins)
    ax[0].set_xticks((bins[:-1] + bins[1:]) / 2, [chr(k) for k in range(ord("A"), ord("J") + 1)])
    ax[0].set_title("Training data")

    bins = np.arange(test_labels.min(), test_labels.max() + 2)
    ax[1].hist(test_labels, bins=bins)
    ax[1].set_xticks((bins[:-1] + bins[1:]) / 2, [chr(k) for k in range(ord("A"), ord("J") + 1)])
    ax[1].set_title("Test data")
    plt.show()


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def balance_check(sizes):
    mean_val = mean(sizes)
    print('mean of # images :', mean_val)
    for i in sizes:
        if abs(i - mean_val) > 0.1 * mean_val:
            print("Too much or less images")
        else:
            print("Well balanced", i)


def load_and_display_pickle(data_sets, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    num_of_images = []
    for pickle_file in data_sets:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            print('Total images in', pickle_file, ':', len(data))

            for index, img in enumerate(data):
                if index == sample_size:
                    break
                ax = fig.add_subplot(len(data_sets), sample_size, sample_size * data_sets.index(pickle_file) +
                                     index + 1)
                ax.imshow(img)
                ax.set_axis_off()
                ax.imshow(img)

            num_of_images.append(len(data))

    balance_check(num_of_images)
    plt.show()
    return num_of_images


def display_overlap(overlap, source_data_set, target_data_set):
    overlap = {k: v for k, v in overlap.items() if len(v) >= 3}
    item = random.choice(list(overlap.keys()))
    images = np.concatenate(([source_data_set[item]], target_data_set[overlap[item][0:7]]))
    plt.suptitle(item)
    for i, img in enumerate(images):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.show()


def display_sample_data_set(data_set, labels, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    items = random.sample(range(len(labels)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[item]))
        plt.imshow(data_set[item])
    plt.show()