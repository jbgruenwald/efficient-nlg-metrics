Can we do that simpler?
===
Simple, Efficient, High-Quality Evaluation Metrics for NLG
---

This repository contains code for calculating and evaluating various efficient nlg metrics. It is structured as follows:

- Folder **experiments** contains code for the experiments conducted
  - the script for evaluating metrics is experiments/evaluate-metrics.py
  - after evaluating, the results will can be found in text files in **results** folder+
  - for averaging the results and writing them into the Tex tables and figures, experiments/process_results/process_results.py can be used
- Folder **metrics** contains code for calculation of the metrics (this code originates from their respective repositories)
- Folder **datasets** contains data for training and evaluation