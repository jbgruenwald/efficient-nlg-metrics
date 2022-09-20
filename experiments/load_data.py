import os

data_path = '../datasets/'
data = { 'wmt15': {}, 'wmt16': {}, 'wmt20': {}, 'wmt21': {} }

def load_data(data, split_data=0):
    print('loading data')

    # wmt15
    language_pairs = ['de-en', 'cs-en', 'fi-en', 'ru-en']
    srcs = []
    hyps = []
    refs = []
    gold = []
    langs = []
    for language_pair in language_pairs:
        with open('%swmt15-newstest/DAseg.newstest2015.source.%s' % (data_path, language_pair)) as f:
            srcs += [line.strip() for line in f]
        with open('%swmt15-newstest/DAseg.newstest2015.mt-system.%s' % (data_path, language_pair)) as f:
            hyps += [line.strip() for line in f]
        with open('%swmt15-newstest/DAseg.newstest2015.reference.%s' % (data_path, language_pair)) as f:
            refs += [line.strip() for line in f]
        with open('%swmt15-newstest/DAseg.newstest2015.human.%s' % (data_path, language_pair)) as f:
            gold += [float(line.strip()) for line in f]
        langs += [language_pair]*(len(gold)-len(langs))
    if split_data != 0:
        srcs = srcs[:split_data]
        hyps = hyps[:split_data]
        refs = refs[:split_data]
        gold = gold[:split_data]
        langs = langs[:split_data]
    data['wmt15'] = { 'srcs': srcs, 'hyps': hyps, 'refs': refs, 'gold': gold, 'langs': langs }

    # wmt16
    language_pairs = ['de-en', 'cs-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
    srcs = []
    hyps = []
    refs = []
    gold = []
    langs = []
    for language_pair in language_pairs:
        with open('%swmt16-selection/DAseg.newstest2016.source.%s' % (data_path, language_pair)) as f:
            srcs += [line.strip() for line in f]
        with open('%swmt16-selection/DAseg.newstest2016.mt-system.%s' % (data_path, language_pair)) as f:
            hyps += [line.strip() for line in f]
        with open('%swmt16-selection/DAseg.newstest2016.reference.%s' % (data_path, language_pair)) as f:
            refs += [line.strip() for line in f]
        with open('%swmt16-selection/DAseg.newstest2016.human.%s' % (data_path, language_pair)) as f:
            gold += [float(line.strip()) for line in f]
        langs += [language_pair]*(len(gold)-len(langs))
    if split_data != 0:
        srcs = srcs[:split_data]
        hyps = hyps[:split_data]
        refs = refs[:split_data]
        gold = gold[:split_data]
        langs = langs[:split_data]
    data['wmt16'] = { 'srcs': srcs, 'hyps': hyps, 'refs': refs, 'gold': gold, 'langs': langs }

    # wmt20
    hyps_dir = '%swmt20-selection/zhen-outputs/' % data_path
    hyps = []
    with open('%swmt20-selection/newstest2020-zhen-src.zh.txt' % data_path) as f:
        srcs = [line.strip() for line in f] *2
    for filename in sorted(os.listdir(hyps_dir)):          # iterate over all the files in the directory
        file = os.path.join(hyps_dir, filename)
        with open(file) as f:
            hyps += [line.strip() for line in f]
    with open('%swmt20-selection/newstest2020-zhen-ref.en.txt' % data_path) as f:
        refs = [line.strip() for line in f] *2
    with open('%swmt20-selection/mqm_newstest2020_zhen.avg_seg_scores.tsv' % data_path) as f:
        gold = [float(line.strip().split()[1]) for line in f.readlines()[1:]]
        langs = ['zh-en']*len(gold)
    if split_data != 0:
        srcs = srcs[:split_data]
        hyps = hyps[:split_data]
        refs = refs[:split_data]
        gold = gold[:split_data]
        langs = langs[:split_data]
    data['wmt20'] = { 'srcs': srcs, 'hyps': hyps, 'refs': refs, 'gold': gold, 'langs': langs }

    # wmt21
    hyps_dir = '%swmt21-selection/newstest-zh-en/' % data_path
    hypst = []
    with open('%swmt21-selection/newstest2021.zh-en.src.zh' % data_path) as f:
        srcst = [line.strip() for line in f] *5
    for filename in sorted(os.listdir(hyps_dir)):          # iterate over all the files in the directory
        file = os.path.join(hyps_dir, filename)
        with open(file) as f:
            hypst += [line.strip() for line in f]
    with open('%swmt21-selection/newstest2021.zh-en.ref.ref-A.en' % data_path) as f:
        refst = [line.strip() for line in f] *5
    with open('%swmt21-selection/mqm-newstest2021_zhen.avg_seg_scores.tsv' % data_path) as f:
        goldt = [line.strip().split()[1] for line in f.readlines()[1:]]
        langst = ['zh-en']*len(goldt)

    # None aussortieren
    srcs = []
    hyps = []
    refs = []
    gold = []
    langs = []
    for src, hyp, ref, gol, lang in zip(srcst, hypst, refst, goldt, langst):
        if gol != 'None':
            srcs.append(src)
            hyps.append(hyp)
            refs.append(ref)
            gold.append(float(gol))
            langs.append(lang)

    if split_data != 0:
        srcs = srcs[:split_data]
        hyps = hyps[:split_data]
        refs = refs[:split_data]
        gold = gold[:split_data]
        langs = langs[:split_data]
    data['wmt21'] = { 'srcs': srcs, 'hyps': hyps, 'refs': refs, 'gold': gold, 'langs': langs }