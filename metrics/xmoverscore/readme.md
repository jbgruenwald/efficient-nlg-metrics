This directory contains a modified version of XMoverScore (https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation).

Two bugs were fixed:
- scorer.py: line 3 was changed to import from .score_utils_2 instead of score_utils_2
- scorer.py: line 30 device was passed into calculation of perplexity
- score_utils_2.py: in line 39 the parameter return_dict=False was added to the call of model()
  (for more details: https://stackoverflow.com/questions/65132144/bertmodel-transformers-outputs-string-instead-of-tensor)