import sys
import os
from urllib.request import urlretrieve
import tarfile
import pickle
from scipy import ndimage
import numpy as np

last_percent_reported = None


def download_progress_hook(count, block_size, total_size):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False,
                   data_root='.', url=''):
    """Download a file if not present, and make sure it's the right size."""
    destination_filename = os.path.join(data_root, filename)

    if force or not os.path.exists(destination_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, destination_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    stats_info = os.stat(destination_filename)
    if stats_info.st_size == expected_bytes:
        print('Found and verified', destination_filename)
    else:
        raise Exception(
            'Failed to verify ' + destination_filename + '. Can you get to it with a browser?')
    return destination_filename


def maybe_extract(filename, force=False, data_root='.', num_classes=10):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images, image_size=0, pixel_depth=0):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    data_set = np.ndarray(shape=(len(image_files), image_size, image_size),
                          dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            data_set[num_images, :, :] = image_data
            num_images += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    data_set = data_set[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full data set tensor:', data_set.shape)
    print('Mean:', np.mean(data_set))
    print('Standard deviation:', np.std(data_set))
    return data_set


def maybe_pickle(data_folders, min_num_images_per_class, force=False, pixel_depth=0, image_size=0):
    data_set_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        data_set_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            data_set = load_letter(folder, min_num_images_per_class,
                                   pixel_depth=pixel_depth, image_size=image_size)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return data_set_names
