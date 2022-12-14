from __future__ import absolute_import, division, print_function
import time
import numpy as np
import torch
import string
from pyemd import emd, emd_with_flow
from torch import nn
from math import log, sqrt
from itertools import chain

from collections import defaultdict, Counter
from torch.multiprocessing import Pool, set_start_method
from functools import partial
try:
    set_start_method('spawn')
except RuntimeError:
    pass


from transformers import AutoTokenizer, AutoModel

class MoverScorer:
    def __init__(self, model_name, distance='wmd', device='cuda'):
        self.model_name = model_name
        self.distance = distance
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.model.to(device)
        self.model.eval()


    def truncate(self, tokens):
        if len(tokens) > self.tokenizer.model_max_length - 2:
            tokens = tokens[0:(self.tokenizer.model_max_length - 2)]
        return tokens

    def process(self, a):
        a = ["[CLS]"]+self.truncate(self.tokenizer.tokenize(a))+["[SEP]"]
        a = self.tokenizer.convert_tokens_to_ids(a)
        return set(a)


    def get_idf_dict(self, arr, nthreads=4):
        idf_count = Counter()
        num_docs = len(arr)

        process_partial = partial(self.process)

        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

        idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
        idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def padding(self, arr, pad_token, dtype=torch.long):
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens, mask

    def bert_encode(self, model, x, attention_mask):
        model.eval()
        with torch.no_grad():
            result = model(x, attention_mask = attention_mask)
        if self.model_name == 'distilbert-base-uncased':
            return result[1]
        else:
            return result[2]

    #with open('stopwords.txt', 'r', encoding='utf-8') as f:
    #    stop_words = set(f.read().strip().split(' '))

    def collate_idf(self, arr, tokenize, numericalize, idf_dict,
                    pad="[PAD]"):

        tokens = [["[CLS]"]+self.truncate(tokenize(a))+["[SEP]"] for a in arr]
        arr = [numericalize(a) for a in tokens]

        idf_weights = [[idf_dict[i] for i in a] for a in arr]

        pad_token = numericalize([pad])[0]

        padded, lens, mask = self.padding(arr, pad_token, dtype=torch.long)
        padded_idf, _, _ = self.padding(idf_weights, pad_token, dtype=torch.float)

        padded = padded.to(device=self.device)
        mask = mask.to(device=self.device)
        lens = lens.to(device=self.device)
        return padded, padded_idf, lens, mask, tokens

    def get_bert_embedding(self, all_sens, model, tokenizer, idf_dict,
                           batch_size=-1):

        padded_sens, padded_idf, lens, mask, tokens = self.collate_idf(all_sens,
                                                          tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                          idf_dict)

        if batch_size == -1: batch_size = len(all_sens)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(all_sens), batch_size):
                batch_embedding = self.bert_encode(model, padded_sens[i:i+batch_size],
                                              attention_mask=mask[i:i+batch_size])
                batch_embedding = torch.stack(batch_embedding)
                embeddings.append(batch_embedding)
                del batch_embedding

        total_embedding = torch.cat(embeddings, dim=-3)
        return total_embedding, lens, mask, padded_idf, tokens

    def _safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-30)

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.baddbmm(
            x2_norm.transpose(-2, -1),
            x1,
            x2.transpose(-2, -1),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
        return res

    def word_mover_score(self, refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords = True, batch_size=256):
        preds = []
        time_a, time_b, time_c = [], [], []
        for batch_start in range(0, len(refs), batch_size):
            batch_refs = refs[batch_start:batch_start+batch_size]
            batch_hyps = hyps[batch_start:batch_start+batch_size]

            time_0 = time.time()
            ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding(batch_refs, self.model, self.tokenizer, idf_dict_ref)
            hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding(batch_hyps, self.model, self.tokenizer, idf_dict_hyp)

            ref_embedding = ref_embedding[-1]
            hyp_embedding = hyp_embedding[-1]

            batch_size = len(ref_tokens)
            for i in range(batch_size):
                ref_ids = [k for k, w in enumerate(ref_tokens[i])
                                    if w in stop_words or '##' in w
                                    or w in set(string.punctuation)]
                hyp_ids = [k for k, w in enumerate(hyp_tokens[i])
                                    if w in stop_words or '##' in w
                                    or w in set(string.punctuation)]

                ref_embedding[i, ref_ids,:] = 0
                hyp_embedding[i, hyp_ids,:] = 0

                ref_idf[i, ref_ids] = 0
                hyp_idf[i, hyp_ids] = 0

            time_a.append(time.time() - time_0)

            if self.distance=='wmd':
                raw = torch.cat([ref_embedding, hyp_embedding], 1)

                raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

                distance_matrix = self.batched_cdist_l2(raw, raw).double().cpu().numpy()
            elif self.distance=='rwmd':
                raw_ref = torch.cat([ref_embedding], 1)
                raw_hyp = torch.cat([hyp_embedding], 1)
                raw_ref.div_(torch.norm(raw_ref, dim=-1).unsqueeze(-1) + 1e-30)
                raw_hyp.div_(torch.norm(raw_hyp, dim=-1).unsqueeze(-1) + 1e-30)

                distance_matrix = self.batched_cdist_l2(raw_ref, raw_hyp).double().cpu().numpy()

            time_b.append(time.time() - time_0 - time_a[-1])

            for i in range(batch_size):
                if self.distance=='wcd':
                    centroid1 = torch.mean(ref_embedding[i], 0).cpu().numpy()
                    centroid2 = torch.mean(hyp_embedding[i], 0).cpu().numpy()
                    raw1 = torch.tensor([[centroid1]])
                    raw2 = torch.tensor([[centroid2]])
                    raw1.div_(torch.norm(raw1, dim=-1).unsqueeze(-1) + 1e-30)
                    raw2.div_(torch.norm(raw2, dim=-1).unsqueeze(-1) + 1e-30)
                    distance_matrix = self.batched_cdist_l2(raw1, raw2).double().cpu().numpy()
                    score = -distance_matrix[0][0][0]
                elif self.distance=='rwmd':
                    dst = distance_matrix[i]
                    dst_t = list(map(list, zip(*dst))) # transposed distance matrix
                    #w_ref = ref_idf[i]
                    #w_hyp = hyp_idf[i]            # scores didn't change with idf weighting
                    left_sum = 0
                    for wi in dst:                 # wi, weight in zip(dst, w_ref):
                        left_sum += min(wi)        # * weight
                    right_sum = 0
                    for wj in dst_t:               # wj, weight in zip(dst_t, w_hyp):
                        right_sum += min(wj)       # * weight
                    score = -max(left_sum, right_sum)
                else:            # wmd
                    c1 = np.zeros(raw.shape[1], dtype=np.float)
                    c2 = np.zeros(raw.shape[1], dtype=np.float)
                    c1[:len(ref_idf[i])] = ref_idf[i]
                    c2[len(ref_idf[i]):] = hyp_idf[i]

                    c1 = self._safe_divide(c1, np.sum(c1))
                    c2 = self._safe_divide(c2, np.sum(c2))

                    dst = distance_matrix[i]
                    _, flow = emd_with_flow(c1, c2, dst)
                    flow = np.array(flow, dtype=np.float32)
                    score = 1 - np.sum(flow * dst)
                preds.append(score)

            time_c.append(time.time() - time_0 - time_a[-1] - time_b[-1])

        time_data = {'A': time_a, 'B': time_b, 'C': time_c}

        return preds, time_data

    import matplotlib.pyplot as plt

    def plot_example(self, is_flow, reference, translation, device='cuda:0'):

        idf_dict_ref = defaultdict(lambda: 1.)
        idf_dict_hyp = defaultdict(lambda: 1.)

        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = self.get_bert_embedding([reference], self.model, self.tokenizer, idf_dict_ref)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = self.get_bert_embedding([translation], self.model, self.tokenizer, idf_dict_hyp)

        ref_embedding = ref_embedding[-1]
        hyp_embedding = hyp_embedding[-1]

        raw = torch.cat([ref_embedding, hyp_embedding], 1)
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)

        distance_matrix = self.batched_cdist_l2(raw, raw)
        masks = torch.cat([ref_masks, hyp_masks], 1)
        masks = torch.einsum('bi,bj->bij', (masks, masks))
        distance_matrix = masks * distance_matrix


        i = 0
        c1 = np.zeros(raw.shape[1], dtype=np.float)
        c2 = np.zeros(raw.shape[1], dtype=np.float)
        c1[:len(ref_idf[i])] = ref_idf[i]
        c2[len(ref_idf[i]):] = hyp_idf[i]

        c1 = self._safe_divide(c1, np.sum(c1))
        c2 = self._safe_divide(c2, np.sum(c2))

        dst = distance_matrix[i].double().cpu().numpy()

        if is_flow:
            _, flow = emd_with_flow(c1, c2, dst)
            new_flow = np.array(flow, dtype=np.float32)
            res = new_flow[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))]
        else:
            res = 1./(1. + dst[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))])

        r_tokens = ref_tokens[i]
        h_tokens = hyp_tokens[i]

        fig, ax = self.plt.subplots(figsize=(len(r_tokens)*0.8, len(h_tokens)*0.8))
        im = ax.imshow(res, cmap='Blues')

        ax.set_xticks(np.arange(len(h_tokens)))
        ax.set_yticks(np.arange(len(r_tokens)))

        ax.set_xticklabels(h_tokens, fontsize=10)
        ax.set_yticklabels(r_tokens, fontsize=10)
        self.plt.xlabel("System Translation", fontsize=14)
        self.plt.ylabel("Human Reference", fontsize=14)
        self.plt.title("Flow Matrix", fontsize=14)

        self.plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    #    for i in range(len(r_tokens)):
    #        for j in range(len(h_tokens)):
    #            text = ax.text(j, i, '{:.2f}'.format(res[i, j].item()),
    #                           ha="center", va="center", color="k" if res[i, j].item() < 0.6 else "w")
        fig.tight_layout()
        self.plt.show()
