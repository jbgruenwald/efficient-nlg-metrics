pred_file = "../../../results/efficient-transformer-metrics/env1-cpu/xmoverscore/scores/xmoverscore-mbert-2m-gpt2-wmt15-env1-cpu-1646085166.5484385.txt"
pred2_file = "../../../results/efficient-transformer-metrics/env1-cpu/xmoverscore/scores/xmoverscore-mbert-2m-tinygpt2-wmt15-env1-cpu-1646085232.1004004.txt"
output_file = "../../../thesis/data/wmt15-xm-tinygpt2.csv"

with open(pred_file) as f:
    pred = [float(line.strip()) for line in f]
with open(pred2_file) as f:
    human = [float(line.strip()) for line in f]
# if 'wmt21' in output_file:
#     with open("../datasets/wmt21-selection/mqm-newstest2021_zhen.avg_seg_scores.tsv") as f:
#         human = [float(line.strip().split()[1]) for line in f.readlines()[1:] if 'None' not in line]
# elif 'wmt16' in output_file:
#     human = []
#     for language_pair in ['de-en', 'cs-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']:
#         with open('../datasets/wmt16-selection/DAseg.newstest2016.human.' + language_pair) as f:
#             human += [float(line.strip()) for line in f]
# else: #wmt15
#     human = []
#     for language_pair in ['de-en', 'cs-en', 'fi-en', 'ru-en']:
#         with open('../datasets/wmt15-newstest/DAseg.newstest2015.human.' + language_pair) as f:
#             human += [float(line.strip()) for line in f]

data = ['%s,%s' % (x, y) for x, y in zip (human, pred)]
with open(output_file, 'a') as f:
    f.write('human,pred\n' + '\n'.join(data))