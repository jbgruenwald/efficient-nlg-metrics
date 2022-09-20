from metrics.sentsim.SentSimScorer import SentSimScorer
from experiments.config import model_name

def sentsim(srcs, hyps, refs, model, sentmodel, distance, langs, batch_size, device):
    if sentmodel == '':
        sentmodel = 'sentdefault'
    scorer = SentSimScorer(wordemb_model=model_name[model], sentemb_model=model_name[sentmodel], embedding_batch_size=batch_size, device=device)
    score = scorer.score(srcs, hyps)
    print(score)
    return {'score': score}

def sentsimwmd(srcs, hyps, refs, model, sentmodel, distance, langs, batch_size, device):
    if sentmodel == '':
        sentmodel = 'sentdefault'
    scorer = SentSimScorer(wordemb_model=model_name[model], sentemb_model=model_name[sentmodel], embedding_batch_size=batch_size, device=device, use_wmd=True)
    score = scorer.score(srcs, hyps)
    print(score)
    return {'score': score}