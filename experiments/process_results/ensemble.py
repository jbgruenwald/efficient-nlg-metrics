from experiments.load_data import load_data
import matplotlib.pyplot as plt
from statistics import mean
import scipy.stats
import numpy as np

data = { 'wmt15': {}, 'wmt16': {}, 'wmt20': {}, 'wmt21': {} }
load_data(data, 0)

scores = {
    'bertscore-bt': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },  # bert-tiny
    'bertscore-tb': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },  # tinybert4
    'bertscore-db': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },  # distilbert
    'moverscore-wmd': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },
    'moverscore-rwmd': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },
    'bartscore': { 'wmt15': [], 'wmt16': [], 'wmt21': [] },
    'comet': { 'wmt15': [], 'wmt16': [], 'wmt21': [] }
}

def normalize(scores):
    mi = min(scores)
    ma = max(scores)
    return [(s-mi)/(ma-mi) for s in scores]

def standardize(scores):
    me = mean(scores)
    dev = np.std(scores)
    return [ (s-me)/dev for s in scores]


with open('../results/efficient-transformer-metrics/env3-cpu/bertscore/scores/bertscore-bert-tiny-wmt15-env3-cpu-1646618970.4440305.txt') as f:
    scores['bertscore-bt']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env3-cpu/bertscore/scores/bertscore-bert-tiny-wmt16-env3-cpu-1646619705.5657105.txt') as f:
    scores['bertscore-bt']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env3-cpu/bertscore/scores/bertscore-bert-tiny-wmt21-env3-cpu-1646620586.699753.txt') as f:
    scores['bertscore-bt']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-tinybert-wmt15-env1-cpu-1648751827.01975.txt') as f:
    scores['bertscore-tb']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-tinybert-wmt16-env1-cpu-1645488320.9791691.txt') as f:
    scores['bertscore-tb']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-tinybert-wmt21-env1-cpu-1645493028.3570535.txt') as f:
    scores['bertscore-tb']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-distilbert-wmt15-env1-cpu-1644180343.7079535.txt') as f:
    scores['bertscore-db']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-distilbert-wmt16-env1-cpu-1644180688.160324.txt') as f:
    scores['bertscore-db']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-distilbert-wmt21-env1-cpu-1644083874.5662365.txt') as f:
    scores['bertscore-db']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-wmd-tinybert-wmt15-env1-cpu-1651845360.5281658.txt') as f:
    scores['moverscore-wmd']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-wmd-tinybert-wmt16-env1-cpu-1645490281.0132494.txt') as f:
    scores['moverscore-wmd']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-wmd-tinybert-wmt21-env1-cpu-1645494687.3135862.txt') as f:
    scores['moverscore-wmd']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-rwmd-tinybert-wmt15-env1-cpu-1653815541.3666563.txt') as f:
    scores['moverscore-rwmd']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-rwmd-tinybert-wmt16-env1-cpu-1653815691.4066489.txt') as f:
    scores['moverscore-rwmd']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/moverscore/scores/moverscore-rwmd-tinybert-wmt21-env1-cpu-1653815887.4765441.txt') as f:
    scores['moverscore-rwmd']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env3-cpu/bartscore/scores/bartscore-distilbart66-wmt15-env3-cpu-1646803137.5317678.txt') as f:
    scores['bartscore']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env3-cpu/bartscore/scores/bartscore-distilbart66-wmt16-env3-cpu-1646807946.9977677.txt') as f:
    scores['bartscore']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env3-cpu/bartscore/scores/bartscore-distilbart66-wmt21-env3-cpu-1646814014.90441.txt') as f:
    scores['bartscore']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

with open('../results/efficient-transformer-metrics/env1-cpu/comet/scores/comet-mminilm12-refbased-5-wmt15-env1-cpu-1649100185.8276813.txt') as f:
    scores['comet']['wmt15'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/comet/scores/comet-mminilm12-refbased-5-wmt16-env1-cpu-1649104495.789963.txt') as f:
    scores['comet']['wmt16'] = standardize(normalize([float(line.strip()) for line in f]))
with open('../results/efficient-transformer-metrics/env1-cpu/comet/scores/comet-mminilm12-refbased-5-wmt21-env1-cpu-1649110291.424165.txt') as f:
    scores['comet']['wmt21'] = standardize(normalize([float(line.strip()) for line in f]))

set = 'wmt21'
correlation = scipy.stats.pearsonr([x+y+z for x,y,z in zip(scores['bertscore-tb'][set], scores['moverscore-wmd'][set], scores['comet'][set])], data[set]['gold'])[0]
print('Correlation:', correlation)