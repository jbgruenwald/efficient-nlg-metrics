from experiments.call_metrics.moverscore import moverscore
from experiments.call_metrics.xmoverscore import xmoverscore
from load_data import load_data
import pandas as pd
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

split_data = 0
batch_size = 1

env_string = 'env1'

device_config = {
    'cpu': 'cpu',
    'gpu': 'cuda:0'
}

data = { 'wmt15': {}, 'wmt16': {}, 'wmt20': {}, 'wmt21': {} }
# which metrics can change their distance algorithm?
metrics_with_distances = [moverscore, xmoverscore]

def main():
    load_data(data, split_data)

    # configuration
    metrics = [xmoverscore]
    models = ['mbert']
    device = 'cpu'
    distances = ['wmd', 'wcd', 'rwmd']
    repeat = 2

    for i in range(repeat):
        for dataset in ['wmt15', 'wmt16', 'wmt21']:
            for metric in metrics:
                for model in models:
                    for distance in distances:
                        # print info on metric and distance
                        print('--------------------')
                        metric_string = metric.__name__+'-'+distance if metric in metrics_with_distances else metric.__name__
                        print('start step measuring with %s on %s using %s' % (metric_string, dataset, device))
                        model_string = model

                        # do inference
                        scores, time_data = metric(data[dataset]['srcs'], data[dataset]['hyps'], data[dataset]['refs'], model, '', distance, data[dataset]['langs'], batch_size, device_config[device], timesteps=True)

                        df = pd.DataFrame(time_data)

                        # read last output for concatenation
                        output_directory = '../results/metric-steps/%s-%s/%s' % (env_string, device, metric.__name__)
                        filename = output_directory + '/%s-%s-%s-%s.csv' % (metric_string, model_string, env_string, device)
                        if os.path.isfile(filename):                # if file exists
                            df_old = pd.read_csv(filename)          # read the old output
                            df_new = pd.concat([df_old, df])        # and concat the new output
                        else:
                            df_new = df                             # otherwise just use only the new output
                        print(df_new)

                        print(df_new.mean(axis=0))                  # print an average of each column of the whole data

                        # write data
                        if not os.path.exists(output_directory):    # if directory doesn't exist yet
                            os.makedirs(output_directory)           # -> create it
                        df_new.to_csv(filename, index=False)        # save new data

if __name__ == "__main__":
    main()