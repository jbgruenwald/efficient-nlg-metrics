Efficient Evaluation Metrics for Natural Language Generation
===
Master Thesis by Jens Grünwald
---

This repository contains code and documentation of my master thesis. It is structured as follows:

- Folder **thesis** contains the Tex code for the written thesis, that is handed in at TU Darmstadt
- Folder **experiments** contains code for the experiments conducted in the thesis
  - the script for evaluating metrics is experiments/evaluate-metrics.py
  - after evaluating, the results will can be found in text files in **results** folder+
  - for averaging the results and writing them into the Tex tables and figures, experiments/process_results/process_results.py can be used
- Folder **metrics** contains code for the metrics
- Folder **datasets** contains data for training and evaluation