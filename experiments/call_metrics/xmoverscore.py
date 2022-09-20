import time

from metrics.xmoverscore.scorer import XMOVERScorer
from experiments.config import xmoverscore_original_mapping_folder, xmoverscore_trained_mapping_folder, model_name
import numpy as np
import torch

model2layers = {
    'mbert': 8,
    'xlmr': 12,
    'distilmbert': 6,
    'xtremedistil': 6,
    'mminilm6': 6,
    'mminilm12': 12
}

def xmoverscore(srcs, hyps, refs, model, lm, distance, langs, batch_size, device, timesteps=False):
    # this function can handle data of different language pairs
    # but it expects them to be "ordered" - all samples of one
    # language pair have to occur consecutively. Otherwise the
    # batches of different language pairs are split wrong

    xms_scores = []
    lm_scores = []
    scores = []
    time_data = {'A': [], 'B': [], 'C': [], 'D': []}

    if lm == '':
        lm = 'gpt2'

    # initialize scorer class
    scorer = XMOVERScorer(model_name[model], model_name[lm], False, distance=distance, device=device)

    mapping_layer = model2layers[model]

    # iterate over language pairs
    while len(srcs) > 0:
        # cut language batch
        language_pair = langs[0]                    # take first sample's language pair
        langbatch_size = langs.count(language_pair) # and count how often it occurs in the data
        srcs_langbatch = srcs[:langbatch_size]      # then cut the batch from the data
        hyps_langbatch = hyps[:langbatch_size]      # then cut the batch from the data
        srcs = srcs[langbatch_size:]                # and reduce the remaining data by this langbatch
        hyps = hyps[langbatch_size:]                # and reduce the remaining data by this langbatch

        # calculate new projection mapping
        if model == 'mbert':
            temp = np.load('%s/layer-%s/europarl-v7.%s.2k.%s.BAM' % (xmoverscore_original_mapping_folder, mapping_layer, language_pair, mapping_layer), allow_pickle=True)
        else:
            temp = np.load('%s/%s/WikiMatrix.%s.2k.%s.BAM.map.npy' % (xmoverscore_trained_mapping_folder, model, language_pair, mapping_layer), allow_pickle=True)
        projection = torch.tensor(temp, dtype=torch.float).to(device)

        # calculate new bias mapping
        if model == 'mbert':
            temp = np.load('%s/layer-%s/europarl-v7.%s.2k.%s.GBDD' % (xmoverscore_original_mapping_folder, mapping_layer, language_pair, mapping_layer), allow_pickle=True)
        else:
            temp = np.load('%s/%s/WikiMatrix.%s.2k.%s.GBDD.map.npy' % (xmoverscore_trained_mapping_folder, model, language_pair, mapping_layer), allow_pickle=True)
        bias = torch.tensor(temp, dtype=torch.float).to(device)

        # calculate word mover distance scores and language model scores
        xms_langbatch_scores, langbatch_time_data = scorer.compute_xmoverscore('CLP', projection, bias, srcs_langbatch, hyps_langbatch, bs=batch_size, layer=mapping_layer)
        time_lm_0 = time.time()
        lm_langbatch_scores = scorer.compute_perplexity(hyps_langbatch, bs=1)
        time_lm = [(time.time() - time_lm_0)/langbatch_size] * langbatch_size

        time_data['A'].extend(langbatch_time_data['A'])
        time_data['B'].extend(langbatch_time_data['B'])
        time_data['C'].extend(langbatch_time_data['C'])
        time_data['D'].extend(time_lm)

        # combine scores
        scores.extend(metric_combination(xms_langbatch_scores, lm_langbatch_scores, [1, 0.1]))
        xms_scores += xms_langbatch_scores
        lm_scores += lm_langbatch_scores

    print(scores)
    print(xms_scores)
    print(lm_scores)

    if timesteps:
        return {'score': scores, 'xms_score': xms_scores, 'lm_score': lm_scores}, time_data
    else:
        return {'score': scores, 'xms_score': xms_scores, 'lm_score': lm_scores}

# combines two metrics using the weights in parameter alpha
def metric_combination(a, b, alpha):
    return alpha[0]*np.array(a) + alpha[1]*np.array(b)