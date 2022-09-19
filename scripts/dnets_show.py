#!/usr/bin/env python3

import os
import argparse
import shutil
from dnets import data_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--list', action='store_true', help='show the project')
    parser.add_argument('-g', '--get', dtype=str, help='get the project conf.yaml file')

    args = parser.parse_args()

    projs = []
    for name in os.listdir(data_root):
        projs.append(name)

    if args.list:
        print('id\tname')
        for idx, name in enumerate(projs):
            print("%2d\t%s" % (idx, name))
    
    if args.get:
        try:
            path = os.path.join(data_root, projs[int(args.get)])
            assert os.path.exists(path)
        except:
            path = os.path.join(data_root, args.get)
        shutil.copy(path, './')




