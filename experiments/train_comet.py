import sys
sys.path.insert(1, '..')

from metrics.comet.cli.train import train_command
import time
import os

train_cfgs = [
    "train_metrics/xlmr-refbased-5.yaml",
    "train_metrics/xlmr-refbased-10.yaml",
    "train_metrics/mminilm12-refbased-5.yaml",
    "train_metrics/mminilm12-refbased-10.yaml"
    ]

for cfg in train_cfgs:
    # start time
    start_time = time.time()
    cfg_name = os.path.splitext(os.path.basename(cfg))[0]

    print('--------------------')
    print('start training with %s' % (cfg_name))
    train_command(cfg)

    # calculate runtime
    runtime = time.time() - start_time
    output = 'runtime (sec): %s' % runtime

    print(output)

    # write output
    output_directory = '../results/train-metrics/comet/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(output_directory + '/comet-%s.txt' % (cfg_name), 'a') as f:
        f.write(output + '\n')