from metrics.baryscore.bary_score import BaryScoreMetric
from experiments.config import model_name

def baryscore(srcs, hyps, refs, model, model2, distance, langs, batch_size, device):
    model_layers = {
        'bert': 5,
        'deebert-mnli': 5,
        'roberta-large': 5,
        'distilbert': 5,
        'tinybert': 4,
        'bert-tiny': 2
    }
    metric_call = BaryScoreMetric(model_name=model_name[model], last_layers=model_layers[model], device=device)
    metric_call.prepare_idfs(refs, hyps)

    scores = {
        'score': [],
        'SD_10': [],
        'SD_1': [],
        'SD_5': [],
        'SD_0.1': [],
        'SD_0.5': [],
        'SD_0.01': [],
        'SD_0.001': []
    }
    # iterate over batches
    i = 0
    while len(refs) > 0:
        refs_batch = refs[:min(batch_size, len(refs))]
        hyps_batch = hyps[:min(batch_size, len(hyps))]
        refs = refs[min(batch_size, len(refs)):]
        hyps = hyps[min(batch_size, len(hyps)):]
        scores_batch = metric_call.evaluate_batch(refs_batch, hyps_batch)
        print('batch %i:' % i, scores_batch)
        scores['score'] += [-x for x in scores_batch['baryscore_W']]
        scores['SD_10'] += [-x for x in scores_batch['baryscore_SD_10']]
        scores['SD_1'] += [-x for x in scores_batch['baryscore_SD_1']]
        scores['SD_5'] += [-x for x in scores_batch['baryscore_SD_5']]
        scores['SD_0.1'] += [-x for x in scores_batch['baryscore_SD_0.1']]
        scores['SD_0.5'] += [-x for x in scores_batch['baryscore_SD_0.5']]
        scores['SD_0.01'] += [-x for x in scores_batch['baryscore_SD_0.01']]
        scores['SD_0.001'] += [-x for x in scores_batch['baryscore_SD_0.001']]
        i += 1
    return scores