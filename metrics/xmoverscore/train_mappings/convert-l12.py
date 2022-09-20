#!/usr/bin/env python
from os import listdir
from numpy import loadtxt, save, full


for file in listdir():
    if file.endswith(("GBDD", "BAM")):
        print(file)
        mapping = loadtxt(file)
        if mapping.ndim == 1:
            mapping = full((768, 768), mapping)
        save(f"binary/{file}", mapping, allow_pickle=False)
