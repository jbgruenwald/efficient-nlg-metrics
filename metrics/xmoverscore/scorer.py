from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .score_utils_2 import word_mover_score, lm_perplexity
class XMOVERScorer:

    def __init__(
        self,
        model_name=None,
        lm_name=None,
        do_lower_case=False,
        distance='wmd',
        device='cuda:0'
    ):        

        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.model.to(device)
        
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_name)        
        self.lm.to(device)
        self.device = device

        self.distance = distance

    def compute_xmoverscore(self, mapping, projection, bias, source, translations, ngram=2, bs=32, layer=8, dropout_rate=0.3):
        return word_mover_score(mapping, projection, bias, self.model, self.tokenizer, source, translations, \
                                n_gram=ngram, layer=layer, dropout_rate=dropout_rate, batch_size=bs, distance=self.distance, device=self.device)
                     
    def compute_perplexity(self, translations, bs):        
        return lm_perplexity(self.lm, translations, self.lm_tokenizer, batch_size=bs, device=self.device)
