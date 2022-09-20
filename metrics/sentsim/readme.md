This directory contains a modified version of SentSim.

SentSim was originally released at: https://github.com/Rain9876/Unsupervised-crosslingual-Compound-Method-For-MT

This implementation is based on: https://github.com/potamides/unsupervised-metrics
Scorer was changed:
- doesn't inherit from CommonScore
- score() isn't called by CommonScore.correlation() but from code/sentsim.py, dataset is loaded before
- bertscore is not calculated by datasets.load_metric() but the one in this project (metrics/bert_score)
- bertscore batch size was added