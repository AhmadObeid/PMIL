import os
import sys
sys.path.append(os.path.join(os.path.expanduser('~/../../media/vsap/New Volume1/utilities')))
from utils2 import get_bad
import argparse
import pdb
from distutils.util import strtobool
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', help="bm (benign vs malignant), or NT (Normal vs Tumor)", default='bm',
                        choices=('bm', 'NT'))
    parser.add_argument('--data', help="glas/split_glas or camelyon16/256x256", default='glas/split_glas',
                        choices=("glas/split_glas", "camelyon16/256x256"))
    parser.add_argument('--magn', type=lambda x: bool(strtobool(x)), default=0)
    args = parser.parse_args()
    data_dir = os.path.join(os.path.expanduser('~'), '../../media/vsap/New Volume1/datasets/histopathology')
    rootDir = os.path.join(data_dir, args.data)

    wanted_split = 'train'
    types = ['malignant', 'benign'] if args.types == "bm" else ['Normal', 'Tumor']
    splits = ['train', 'val', 'test']
    magnification = [str(40*(2**-i)) for i in range(7)] if args.magn else "."
    for split in splits:
        for typ in types:
            for magn in magnification[1:]: ##### Currently skipping 40
                directory = os.path.join(rootDir,split, typ, magn, 'rgb')
                start = time.time()
                get_bad(rootDir, directory,"entropy")
                print(f"For {magn}, {typ}, {split}, it took {time.time()-start} seconds")
