#!/usr/bin/env python3

import argparse
from os import listdir
from numpy import loadtxt, save, float32

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, required=True,
                    help="path to directory that is to be converted")
args = parser.parse_args()

path = args.directory

for file in listdir(path):
    if file.endswith(("GBDD.map", "BAM.map")):
        print(file)
        mapping = loadtxt(path+file, dtype=float32)
        save(f"{path}binary/{file}", mapping, allow_pickle=False)
