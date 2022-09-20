import os

dir = "../../results/efficient-transformer-metrics/env3-gpu/transquest"

def rename(dir):
    for filename in os.listdir(dir):          # iterate over all the files in the directory
        file = os.path.join(dir, filename)
        if os.path.isfile(file):
            os.rename(file, os.path.join(dir, filename.replace("quest-mmi", "quest-tq-mmi")))
        else:
            rename(file)

rename(dir)