import pandas as pd
import os

env_string = 'env1'
device = 'cpu'

for metric in ['moverscore', 'xmoverscore']:
    for distance in ['wmd', 'wcd', 'rwmd']:
        # print info for current metric and distance
        print('--------------------')
        print('metric step averages of %s with %s' % (metric, distance))
        model = 'bert' if metric == 'moverscore' else 'mbert'

        # read csv file
        directory = '../../results/metric-steps/%s-%s/%s' % (env_string, device, metric)
        filename = directory + '/%s-%s-%s-%s-%s.csv' % (metric, distance, model, env_string, device)
        if os.path.isfile(filename):            # if file exists
            df = pd.read_csv(filename)          # read the content
            print(df.mean(axis=0))              # and print averages of each column