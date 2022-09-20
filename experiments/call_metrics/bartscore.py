from metrics.bartscore.bart_score import BARTScorer
from experiments.config import model_name, bartscore_pth_file


def bartscore(srcs, hyps, refs, model, model2, distance, langs, batch_size, device):
    if '+pth' in model:
        bart_scorer = BARTScorer(device=device, checkpoint=model_name[model[:-4]])
        bart_scorer.load(path=bartscore_pth_file)
    else:
        bart_scorer = BARTScorer(device=device, checkpoint=model_name[model])
    score = bart_scorer.score(hyps, refs, batch_size=batch_size) # generation scores from the first list of texts to the second list of texts.
    print(score)
    return {'score': score}