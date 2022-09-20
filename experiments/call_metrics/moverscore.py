from metrics.moverscore.moverscore_v2 import MoverScorer
from experiments.config import model_name

def moverscore(srcs, hyps, refs, model, model2, distance, langs, batch_size, device, timesteps=False):
    mover_scorer = MoverScorer(model_name[model], distance=distance, device=device)

    idf_dict_hyp = mover_scorer.get_idf_dict(hyps) # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = mover_scorer.get_idf_dict(refs) # idf_dict_ref = defaultdict(lambda: 1.)

    scores, time_data = mover_scorer.word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True, batch_size=batch_size)
    print(scores)
    if timesteps:
        return {'score': scores}, time_data
    else:
        return {'score': scores}