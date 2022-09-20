import sys
sys.path.insert(1, '..')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # otherwise AutoTokenizer class will stop with a warning during the second run
os.environ['MPLCONFIGDIR'] = '../../mpl_config'

from experiments.call_metrics.bertscore import bertscore
from experiments.call_metrics.moverscore import moverscore
from experiments.call_metrics.baryscore import baryscore
from experiments.call_metrics.bartscore import bartscore
from experiments.call_metrics.xmoverscore import xmoverscore
from experiments.call_metrics.sentsim import sentsim
from experiments.call_metrics.frugalscore import frugalscore
from experiments.call_metrics.comet import comet
from experiments.call_metrics.transquest import transquest
from load_data import load_data
from transformers import AutoModel
from config import model_name
import scipy.stats
import torch
import time

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
    setup_before_first_usage()

    load_data(data, split_data)

    ### reference-based metrics

    metrics = [bertscore, moverscore, baryscore]
    models = ['bert', 'bert-tiny', 'distilbert', 'tinybert', 'deebert-mnli']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    # only bertscore needs roberta-large as baseline
    metrics = [bertscore]
    models = ['roberta-large']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    # moverscore with wmd variations
    models = ['bert', 'tinybert']
    distances = ['wcd', 'rwmd']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device, distances=distances)

    # bartscore
    metrics = [bartscore]
    models = ['bart-large-cnn+pth', 'bart-large-cnn', 'bart', 'distilbart66', 'distilbart123', 'distilbart-t2s', 'distilbart-mnli9']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    ### reference-free metrics

    metrics = [xmoverscore, sentsim]
    models = ['mbert', 'xlmr', 'xtremedistil', 'mminilm6', 'mminilm12']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    metrics = [sentsim]
    models = ['distilmbert']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    # xmoverscore language models
    metrics = [xmoverscore]
    models = ['mbert']
    model2s = ['distilgpt2']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device, model2s=model2s)

    # sentsim sentence embedding models
    metrics = [sentsim]
    models = ['xlmr', 'mminilm12']
    model2s = ['use2', 'pminilm12', 'pmpnet']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device, model2s=model2s)

    # xmoverscore with wmd variations
    models = ['mbert']
    distances = ['wcd', 'rwmd']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device, distances=distances)

    metrics = [comet]
    models = [
        'xlmr-refbased-5',
        'xlmr-refbased-10',
        'mminilm12-refbased-5',
        'mminilm12-refbased-10'
    ]
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    metrics = [transquest]
    models = [
        'tq-xlmr-large',
        'tq-xlmr',
        'tq-mminilm12',
        'tq-mminilm6',
        'tq-xlmr-large-adapter',
        'tq-xlmr-adapter',
        'tq-mminilm12-adapter',
        'tq-mminilm6-adapter'
    ]
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)

    ### frugalscore
    metrics = [frugalscore]
    models = ['fs-tiny-b', 'fs-tiny-r', 'fs-tiny-d', 'fs-tiny-ms',
              'fs-small-b', 'fs-small-r', 'fs-small-d', 'fs-small-ms',
              'fs-medium-b', 'fs-medium-r', 'fs-medium-d', 'fs-medium-ms']
    datasets = ['wmt15', 'wmt16', 'wmt21']
    device = 'cpu'
    evaluate_metrics(metrics, models, datasets, device)


def setup_before_first_usage():
    # download models first to prevent distorting time later when durations are measured
    for model in model_name:
        AutoModel.from_pretrained(model_name[model])

    # nltk package punkt needed for xmoverscore
    import nltk
    nltk.download("punkt", download_dir='../../nltk_cache')


def evaluate_metrics(metrics, models, datasets, device, model2s=[''], distances=['wmd']):
    for dataset in datasets:
        for metric in metrics:
            for model in models:
                for model2 in model2s:
                    for distance in distances:
                        # start time
                        start_time = time.time()

                        # do inference
                        print('--------------------')
                        metric_string = metric.__name__+'-'+distance if metric in metrics_with_distances else metric.__name__
                        model_string = model+'-2m-'+model2 if model2 != '' else model
                        print('start inference with %s and %s on %s using %s' % (metric_string, model_string, dataset, device))
                        scores = metric(data[dataset]['srcs'], data[dataset]['hyps'], data[dataset]['refs'], model, model2, distance, data[dataset]['langs'], batch_size, device_config[device])

                        # calculate duration
                        duration = time.time() - start_time
                        output = 'duration (sec): %s' % duration

                        # memory usage
                        if device == 'gpu':
                            peak_memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
                            torch.cuda.reset_peak_memory_stats()
                        else:
                            peak_memory_usage = 0
                        output += '\npeak memory-usage: %s' % peak_memory_usage

                        # evaluate scores
                        pearson = {}
                        for key in scores:
                            pearson[key] = scipy.stats.pearsonr(scores[key], data[dataset]['gold'])[0]

                        print(output)
                        print('\npearson correlation: %s' % pearson)
                        output += '\npearson correlation: %s' % pearson['score']

                        # write output
                        output_directory = '../results/efficient-transformer-metrics/%s-%s/%s' % (env_string, device, metric.__name__)
                        if not os.path.exists(output_directory):
                            os.makedirs(output_directory)
                        with open(output_directory + '/%s-%s-%s-%s-%s.txt' % (metric_string, model_string, dataset, env_string, device), 'a') as f:
                            f.write(output + '\n')

                        # write scores
                        if not os.path.exists(output_directory + '/scores'):
                            os.makedirs(output_directory + '/scores')
                        with open(output_directory + '/scores/%s-%s-%s-%s-%s-%s.txt' % (metric_string, model_string, dataset, env_string, device, time.time()), 'a') as f:
                            f.write('\n'.join([str(float(i)) for i in scores['score']]))

if __name__ == "__main__":
    main()