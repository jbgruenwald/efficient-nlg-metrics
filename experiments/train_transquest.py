import sys
import pandas as pd
from transformers import TrainingArguments, AdapterArguments
sys.path.insert(1, '..')

from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from sklearn.metrics import mean_absolute_error
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from metrics.transquest.run_model import MonoTransQuestAdapterModel
from experiments.train_metrics.transquest.monotransquest_config import monotransquest_config
from experiments.load_data import load_data
import scipy
import torch
import time
import os

env_string = 'env1'

device_config = {
    'cpu': 'cpu',
    'gpu': 'cuda:0'
}

device='cpu'
train_cfgs = [
    {
        'save_name': 'mminilm6',
        'model': 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
        'adapter': False
    },
    {
        'save_name': 'mminilm12',
        'model': 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large',
        'adapter': False
    },
    {
        'save_name': 'xlmr',
        'model': 'xlm-roberta-base',
        'adapter': False
    },
    {
        'save_name': 'xlmr-large',
        'model': 'xlm-roberta-large',
        'adapter': False
    },
    {
        'save_name': 'mminilm6-adapter',
        'model': 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
        'adapter': True
    },
    {
        'save_name': 'mminilm12-adapter',
        'model': 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large',
        'adapter': True
    },
    {
        'save_name': 'xlmr-adapter',
        'model': 'xlm-roberta-base',
        'adapter': True
    },
    {
        'save_name': 'xlmr-large-adapter',
        'model': 'xlm-roberta-large',
        'adapter': True
    },
]

def read_data(f):
    df = pd.read_csv('../datasets/transquest-traindata/2017-scores.' + f + '.csv')
    df = df.drop(columns=['lp', 'ref', 'score', 'annotators'])
    df = df.rename(columns={'src': 'text_a', 'mt': 'text_b', 'raw_score': 'labels'})
    df['labels'] = df['labels'].div(100)
    return df

train_df = read_data('4000.train')
eval_df = read_data('500.test')


data = { 'wmt15': {}, 'wmt16': {}, 'wmt20': {}, 'wmt21': {} }
load_data(data, 0)

for cfg in train_cfgs:
    # start time
    start_time = time.time()

    monotransquest_config['output_dir'] = '../../transquest/'+cfg['save_name']
    monotransquest_config['best_model_dir'] = monotransquest_config['output_dir'] + '/best_model'

    print('--------------------')
    if cfg["adapter"]:
        print('start adapter training with %s' % (cfg["model"]))

        model = MonoTransQuestAdapterModel("xlmroberta", cfg["model"], num_labels=1, use_cuda=(0 if device=='cpu' else 1), args=monotransquest_config)
        model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    else:
        print('start training with %s' % (cfg["model"]))

        model = MonoTransQuestModel("xlmroberta", cfg["model"], num_labels=1, use_cuda=(0 if device=='cpu' else 1), args=monotransquest_config)
        model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)

    # calculate runtime
    runtime = time.time() - start_time
    output = 'runtime (sec): %s' % runtime

    print(output)

    # write output
    output_directory = '../results/train-metrics/transquest/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(output_directory + '/transquest-%s.txt' % (cfg['save_name']), 'a') as f:
        f.write(output + '\n')

    ################################################ evaluation

    print(model.model.roberta.active_adapters)
    for dataset in ['wmt15', 'wmt16', 'wmt21']:
        # start time
        start_time = time.time()

        # do inference
        print('--------------------')
        metric_string = 'transquest'
        model_string = cfg["model"]
        print('start inference with %s and %s on %s using %s' % (metric_string, model_string, dataset, device))
        tq_data = [[src, hyp] for src, hyp in zip(data[dataset]['srcs'], data[dataset]['hyps'])]
        scores, raw_outputs = model.predict(tq_data)
        print(scores)

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
        pearson = scipy.stats.pearsonr(scores, data[dataset]['gold'])[0]

        print(output)
        print('\npearson correlation: %s' % pearson)
        output += '\npearson correlation: %s' % pearson['score']

        # write output
        output_directory = '../results/efficient-transformer-metrics/%s-%s/%s' % (env_string, device, 'transquest')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(output_directory + '/%s-%s-%s-%s-%s.txt' % (metric_string, model_string, dataset, env_string, device), 'a') as f:
            f.write(output + '\n')

        # write scores
        if not os.path.exists(output_directory + '/scores'):
            os.makedirs(output_directory + '/scores')
        with open(output_directory + '/scores/%s-%s-%s-%s-%s-%s.txt' % (metric_string, model_string, dataset, env_string, device, time.time()), 'a') as f:
            f.write('\n'.join([str(float(i)) for i in scores]))