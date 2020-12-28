
import glob
import json
import os
import time
import sys

import numpy as np


if __name__ == '__main__':

    print('building the shard sizes ...')

    path = sys.argv[1]
    print('> reading numpy files from {}'.format(path))

    npy_files = glob.glob(path + '/*.npy')
    npy_files.sort()
    print('  found {} numpy files'.format(len(npy_files)))

    size_dict = {}
    counter = 0
    start_time = time.time()
    for filename in npy_files:
        data = np.load(filename, allow_pickle=True)
        size = np.hstack(data).size
        np_filename = os.path.basename(filename)
        size_dict[np_filename] = size
        counter += 1
        if counter % 10 == 0:
            print('   processed {} files in {:.2f} seconds'.format(
                counter, time.time() - start_time))

    output_filename = os.path.join(path, 'sizes.txt')
    with open(output_filename, 'w') as f:
        json.dump(size_dict, f)
    print('> wrote sizes to {}'.format(output_filename))
