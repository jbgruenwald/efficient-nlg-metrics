from experiments.load_data import load_data
import scipy.stats

data = { 'wmt15': {}, 'wmt16': {}, 'wmt20': {}, 'wmt21': {} }
load_data(data, 10)

print(data['wmt21'])

with open('../../results/efficient-transformer-metrics/env3-cpu/bertscore/scores/bertscore-distilbert-wmt21-env3-cpu-1644953236.5700877.txt') as f:
    scores3 = [float(line.strip()) for line in f]

with open('../../results/efficient-transformer-metrics/env1-cpu/bertscore/scores/bertscore-distilbert-wmt21-env1-cpu-1644083874.5662365.txt') as f:
    scores1 = [float(line.strip()) for line in f]

correlation = scipy.stats.pearsonr(scores1, data['wmt21']['gold'])[0]
print('Correlation:', correlation)