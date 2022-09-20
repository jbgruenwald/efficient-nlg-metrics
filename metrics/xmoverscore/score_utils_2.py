from __future__ import absolute_import, division, print_function
import time
import numpy as np
import torch
from math import sqrt
from pyemd import emd_with_flow

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        output, _, x_encoded_layers, _ = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask, return_dict=False)
    return x_encoded_layers

def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device='cuda:0'):
    tokens = [["[CLS]"]+tokenize(a)+["[SEP]"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens

from collections import defaultdict

def cross_lingual_mapping(mapping, embedding, projection, bias):
    batch_size = embedding.shape[0]
    n_tokens = embedding.shape[1]
    
    if mapping == 'CLP':
        embedding = torch.matmul(embedding, projection)
    if mapping == 'UMD':
        embedding = embedding - (embedding * bias).sum(2, keepdim=True) * bias.repeat(batch_size, n_tokens, 1)        
    return embedding


def lm_perplexity(model, hyps, tokenizer, batch_size=1, device='cuda:0'):
    preds = []        
    model.eval()
    for batch_start in range(0, len(hyps), batch_size):
        batch_hyps = hyps[batch_start:batch_start+batch_size]    
        
        tokenize_input = tokenizer.tokenize(batch_hyps[0])
        
        if len(tokenize_input) <=1:
            preds.append(0)
        else:
            if len(tokenize_input) > 1024:
                tokenize_input = tokenize_input[:1024]
                
            arr = tokenizer.convert_tokens_to_ids(tokenize_input)           
            input_ids = torch.tensor([arr])
            input_ids = input_ids.to(device=device)          
            score = model(input_ids, labels=input_ids)[0]
            preds.append(-score.item())
    return preds

def get_ngram_embs(embeddings, ngram):  
    ngram_embs=[]
    count = 0
    for _ in embeddings[:len(embeddings) - ngram + 1]:  
       ngram_embs.append(embeddings[count:count + ngram, :].mean(0))  
       count = count+1  
    return torch.stack(ngram_embs, 0)

def calc_distance(vec1, vec2):
    return sqrt(sum([abs(v1-v2) for v1, v2 in zip(vec1, vec2)]))

def word_mover_score(mapping, projection, bias, model, tokenizer, src, hyps, \
                     n_gram=2, layer=8, dropout_rate=0.3, batch_size=256, distance='wmd', device='cuda:0'):
    idf_dict_src = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)
    
    preds = []
    time_a, time_b, time_c = [], [], []
    for batch_start in range(0, len(src), batch_size):
        batch_src = src[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]

        time_0 = time.time()
        src_embedding, src_lens, src_masks, src_idf, src_tokens = get_bert_embedding(batch_src, model, tokenizer, idf_dict_src,
                                       device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict_hyp,
                                       device=device)
        
        src_embedding = src_embedding[layer]
        hyp_embedding = hyp_embedding[layer]

        # commented the following block for experiments without remapping
        if type(projection) == tuple and type(bias) == tuple:
            # taking English as a hub language of others
            src_embedding = cross_lingual_mapping(mapping, src_embedding, projection[0], bias[0][0])
            hyp_embedding = cross_lingual_mapping(mapping, hyp_embedding, projection[1], bias[1][0])
        else:
            # mapping non-English to the English space, or the other way round.
            src_embedding = cross_lingual_mapping(mapping, src_embedding, projection, bias[0])
        # end comment
                   
        batch_size = src_embedding.shape[0]

        for i in range(batch_size):   
            src_embedding_i = get_ngram_embs(src_embedding[i, :src_lens[i], :], ngram = n_gram)
            hyp_embedding_i = get_ngram_embs(hyp_embedding[i, :hyp_lens[i], :], ngram = n_gram)

            time_a.append(time.time() - time_0)

            if distance=='wcd':
                time_b.append(time.time() - time_0 - time_a[-1])
                centroid1 = torch.mean(src_embedding_i, 0).cpu().numpy()
                centroid2 = torch.mean(hyp_embedding_i, 0).cpu().numpy()
                W1 = torch.tensor([[centroid1]])
                W2 = torch.tensor([[centroid2]])
                W1.div_(torch.norm(W1, dim=-1).unsqueeze(-1) + 1e-30)
                W2.div_(torch.norm(W2, dim=-1).unsqueeze(-1) + 1e-30)
                distance_matrix = torch.cdist(W1, W2, p=2).double().cpu().numpy()
                score = -distance_matrix[0][0][0]
            elif distance=='rwmd':
                W_src = torch.cat([src_embedding_i], 0)
                W_hyp = torch.cat([hyp_embedding_i], 0)
                W_src.div_(torch.norm(W_src, dim=-1).unsqueeze(-1))
                W_hyp.div_(torch.norm(W_hyp, dim=-1).unsqueeze(-1))
                dist = torch.cdist(W_src, W_hyp, p=2).double().cpu().numpy()

                time_b.append(time.time() - time_0 - time_a[-1])

                left_sum = 0
                for wi in dist:
                    left_sum += min(wi)
                right_sum = 0
                for wj in list(map(list, zip(*dist))):  # transposed distance matrix
                    right_sum += min(wj)
                score = -max(left_sum, right_sum)
            else:            # wmd
                src_idf_i = [1] * (src_lens[i] - n_gram + 1)
                hyp_idf_i = [1] * (hyp_lens[i] - n_gram + 1)

                W = torch.cat([src_embedding_i, hyp_embedding_i], 0)
                W.div_(torch.norm(W, dim=-1).unsqueeze(-1))

                time_b.append(time.time() - time_0 - time_a[-1])

                c1 = list(src_idf_i) + [0] * len(hyp_idf_i)
                c2 = [0] * len(src_idf_i) + list(hyp_idf_i)

                c1 = c1 / np.sum(c1) + 1e-9
                c2 = c2 / np.sum(c2) + 1e-9

                dist = torch.cdist(W, W, p=2).double().cpu().numpy()
                flow = np.stack(emd_with_flow(c1, c2, dist)[1])

                flow = torch.from_numpy(flow[:len(src_idf_i), len(src_idf_i):])
                dist = torch.from_numpy(dist[:len(src_idf_i), len(src_idf_i):])

                # remove noisy elements in a flow
                flow_flatten = flow.reshape(-1)
                idx = torch.nonzero(flow_flatten)
                threshold = flow_flatten[idx].topk(k=max(int(len(idx) * dropout_rate), 1), dim=0, largest=False)[0][-1]
                flow[flow < threshold] = 0

                score = 1 - (flow * dist).sum()
            preds.append(score)
        time_c.append(time.time() - time_0 - time_a[-1] - time_b[-1])

    time_data = {'A': time_a, 'B': time_b, 'C': time_c}
            
    return preds, time_data


