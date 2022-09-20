# download the repository of xmoverscore and reference the "mapping" directory
xmoverscore_original_mapping_folder = '../../ACL20-Reference-Free-MT-Evaluation/mapping'
xmoverscore_trained_mapping_folder = '../metrics/xmoverscore/mappings'

# cache directory for frugalscore models
frugalscore_cache_dir = '../../frugalscore-cache'

# bartscore .pth file
bartscore_pth_file = '../../bart.pth'

# this dictionary maps short names of models (usually used in their papers)
# to their full names on huggingface.com (to indentify and download them)
model_name = {
    ### monolingual models
    'bert': 'bert-base-uncased',
    'roberta-large': 'roberta-large',
    'distilbert': 'distilbert-base-uncased',
    'tinybert': 'huawei-noah/TinyBERT_General_4L_312D',
    'tinybert6': 'huawei-noah/TinyBERT_General_6L_768D',
    'dyntinybert': 'Intel/dynamic_tinybert',
    'dynabertmnli': 'huawei-noah/DynaBERT_MNLI',
    'dynabertsst2': 'huawei-noah/DynaBERT_SST-2',
    'bert-tiny': 'google/bert_uncased_L-2_H-128_A-2',
    'albert': 'albert-base-v2',
    'minilm3': 'nreimers/MiniLM-L3-H384-uncased',
    'minilm6': 'nreimers/MiniLM-L6-H384-uncased',
    'minilm6-384': 'nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large',
    'minilm6-768': 'nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large',
    'minilm12-384': 'nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large',
    'deebert-mnli': 'ji-xin/bert_base-MNLI-two_stage',
    'deeroberta-mnli': 'ji-xin/roberta_base-MNLI-two_stage',
    'deeroberta-mrpc': 'ji-xin/roberta_base-MRPC-two_stage',
    'deeroberta-qnli': 'ji-xin/roberta_base-QNLI-two_stage',
    'deeroberta-sst2': 'ji-xin/roberta_base-SST2-two_stage',
    'deeroberta-qqp': 'ji-xin/roberta_base-QQP-two_stage',
    'deeroberta-rte': 'ji-xin/roberta_base-RTE-two_stage',
    ### models for bart
    'bart': 'facebook/bart-base',
    'bart-large': 'facebook/bart-large',
    'bart-large-cnn': 'facebook/bart-large-cnn',
    # causes an error: 'tiny-mbart': 'sshleifer/tiny-mbart',
    'distilbart66': 'sshleifer/distilbart-cnn-6-6',
    'distilbart126': 'sshleifer/distilbart-cnn-12-6',
    'distilbart123': 'sshleifer/distilbart-cnn-12-3',
    'distilbart-t2s': 'shahrukhx01/distilbart-cnn-12-6-text2sql',
    'distilbartx121': 'sshleifer/distilbart-xsum-12-1',
    'distilbartx123': 'sshleifer/distilbart-xsum-12-3',
    'distilbartx126': 'sshleifer/distilbart-xsum-12-6',
    'distilbart-mnli': 'valhalla/distilbart-mnli-12-6',
    'distilbart-mnli1': 'valhalla/distilbart-mnli-12-1',
    'distilbart-mnli3': 'valhalla/distilbart-mnli-12-3',
    'distilbart-mnli9': 'valhalla/distilbart-mnli-12-9',
    # causes an error: 'mbart-ru': 'IlyaGusev/mbart_ru_sum_gazeta',
    ### multilingual models
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'distilmbert': 'distilbert-base-multilingual-cased',
    'xtremedistil': 'microsoft/xtremedistil-l6-h256-uncased',
    'mminilm6': 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
    'mminilm12': 'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large',# also available on: 'microsoft/Multilingual-MiniLM-L12-H384'
    ### frugalscore models
    'fs-tiny-b': 'moussaKam/frugalscore_tiny_bert-base_bert-score',
    'fs-small-b': 'moussaKam/frugalscore_small_bert-base_bert-score',
    'fs-medium-b': 'moussaKam/frugalscore_medium_bert-base_bert-score',
    'fs-tiny-r': 'moussaKam/frugalscore_tiny_roberta_bert-score',
    'fs-small-r': 'moussaKam/frugalscore_small_roberta_bert-score',
    'fs-medium-r': 'moussaKam/frugalscore_medium_roberta_bert-score',
    'fs-tiny-d': 'moussaKam/frugalscore_tiny_deberta_bert-score',
    'fs-small-d': 'moussaKam/frugalscore_small_deberta_bert-score',
    'fs-medium-d': 'moussaKam/frugalscore_medium_deberta_bert-score',
    'fs-tiny-ms': 'moussaKam/frugalscore_tiny_bert-base_mover-score',
    'fs-small-ms': 'moussaKam/frugalscore_small_bert-base_mover-score',
    'fs-medium-ms': 'moussaKam/frugalscore_medium_bert-base_mover-score',
    ### language models for xmoverscore
    'gpt2': 'gpt2',
    'distilgpt2': 'distilgpt2',
    'tinygpt2': 'sshleifer/tiny-gpt2',
    ### sentmodels for sentsim
    'sentdefault': 'xlm-r-bert-base-nli-stsb-mean-tokens',
    'use1': 'distiluse-base-multilingual-cased-v1',
    'use2': 'distiluse-base-multilingual-cased-v2',
    'pminilm12': 'paraphrase-multilingual-MiniLM-L12-v2',
    'pmpnet': 'paraphrase-multilingual-mpnet-base-v2',
    ### comet models
    'xlmr-reffree-5': '/home/vagrant/lightning_logs/lightning_logs/version_3/checkpoints/epoch=4-step=499.ckpt',
    'xlmr-refbased-5': '/home/vagrant/lightning_logs/lightning_logs/version_9/checkpoints/epoch=4-step=499.ckpt',
    'xlmr-refbased-10': '/home/vagrant/lightning_logs/lightning_logs/version_10/checkpoints/epoch=9-step=999.ckpt',
    'mminilm12-reffree-5': '/home/vagrant/lightning_logs/lightning_logs/version_2/checkpoints/epoch=4-step=499.ckpt',
    'mminilm12-refbased-5': '/home/vagrant/lightning_logs/lightning_logs/version_4/checkpoints/epoch=4-step=499.ckpt',
    'mminilm12-refbased-10': '/home/vagrant/lightning_logs/lightning_logs/version_6/checkpoints/epoch=9-step=999.ckpt',
    ### transquest models
    'tq-xlmr': '../../transquest/xlmr/best_model',
    'tq-xlmr-large': '/home/vagrant/transquest/xlmr-large/best_model',
    'tq-mminilm6': '/home/vagrant/transquest/mminilm6/best_model',
    'tq-mminilm12': '/home/vagrant/transquest/mminilm12/best_model',
    'tq-xlmr-d': '/home/vagrant/transquest/xlmr-d/best_model',
    'tq-mminilm6-d': '/home/vagrant/transquest/mminilm6-d/best_model',
    'tq-mminilm12-d': '/home/vagrant/transquest/mminilm12-d/best_model',
    'tq-xlmr-adapter': '/home/vagrant/transquest/xlmr-adapter/best_model',
    'tq-xlmr-large-adapter': '/home/vagrant/transquest/xlmr-large-adapter/best_model',
    'tq-mminilm6-adapter': '/home/vagrant/transquest/mminilm6-adapter/best_model',
    'tq-mminilm12-adapter': '/home/vagrant/transquest/mminilm12-adapter/best_model',
}