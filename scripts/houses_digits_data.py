from utils import file_utils

url = 'http://ufldl.stanford.edu/housenumbers/'
file_names = [{'name': 'train.tar.gz', 'expected_bytes': 385 * 1024 * 1024},
              {'name': 'test.tar.gz', 'expected_bytes': 264 * 1024 * 1024},
              {'name': 'extra.tar.gz', 'expected_bytes': 1800 * 1024 * 1024}]

for f in file_names:
    file_utils.maybe_download(f.get('name'), f.get('expected_bytes'), url=url)
