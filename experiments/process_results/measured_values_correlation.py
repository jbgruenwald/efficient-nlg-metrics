from process_results import load_results
from statistics import mean
import pandas as pd
import scipy.stats

method = 2  # choose calculation method. for explanation see thesis section 5.1

def main(method=1):
    runtime_cpu = []
    runtime_gpu = []
    memory_gpu = []
    params = []
    diskspace = []
    correlation = []
    correlations = {
        'wmt15': [],
        'wmt16': [],
        'wmt21': []
    }

    metric = 'bertscore'
    models = ['roberta-large', 'bert', 'bert-tiny', 'distilbert', 'tinybert', 'deebert-mnli']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([355359744, 109482240, 4385920, 66362880, 14350248, 109482240])
    diskspace.append([1330, 420, 16.9, 256, 59.8, 503])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'moverscore'
    models = ['bert', 'bert-tiny', 'distilbert', 'tinybert', 'deebert-mnli']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models, distance='-wmd')
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([109482240, 4385920, 66362880, 14350248, 109482240])
    diskspace.append([420, 16.9, 256, 59.8, 503])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'baryscore'
    models = ['bert', 'bert-tiny', 'distilbert', 'tinybert', 'deebert-mnli']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([109482240, 4385920, 66362880, 14350248, 109482240])
    diskspace.append([420, 16.9, 256, 59.8, 503])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'bartscore'
    models = ['bart-large-cnn+pth', 'bart-large-cnn', 'bart', 'distilbart66', 'distilbart123', 'distilbart-t2s', 'distilbart-mnli9']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([406290432, 406290432, 139420416, 229933056, 255120384, 305511424, 355901440])
    diskspace.append([3136, 1510, 532, 439, 973, 1140, 1330])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'xmoverscore'
    models = ['mbert', 'xlmr', 'xtremedistil', 'mminilm6', 'mminilm12']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models, distance='-wmd')
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([177853440, 278043648, 12750080, 106993920, 117640704])
    diskspace.append([681, 1040, 48.7, 205, 225])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'sentsim'
    models = ['xlmr', 'mbert', 'distilmbert', 'xtremedistil', 'mminilm6', 'mminilm12']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([278043648, 177853440, 134734080, 12750080, 106993920, 117640704])
    diskspace.append([1040, 681, 517, 48.7, 205, 225])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'comet'
    models = ['xlmr-refbased-5', 'xlmr-refbased-10', 'mminilm12-refbased-5', 'mminilm12-refbased-10']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([278043648, 278043648, 117640704, 117640704])
    diskspace.append([1040, 1040, 225, 225])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'transquest'
    models = ['tq-xlmr-large-d', 'tq-xlmr-d', 'tq-mminilm12-d', 'tq-mminilm6-d']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([560941057, 278635009, 117788929, 107142145])
    diskspace.append([2244, 1115, 471, 429])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    metric = 'frugalscore'
    models = ['fs-tiny-b', 'fs-tiny-r', 'fs-tiny-d', 'fs-tiny-ms',
              'fs-small-b', 'fs-small-r', 'fs-small-d', 'fs-small-ms',
              'fs-medium-b', 'fs-medium-r', 'fs-medium-d', 'fs-medium-ms']
    t_cpu, t_gpu, m_gpu, r, r_15, r_16, r_21 = load_avg_results(metric, models)
    runtime_cpu.append(t_cpu)
    runtime_gpu.append(t_gpu)
    memory_gpu.append(m_gpu)
    params.append([4385920, 4385920, 4385920, 4385920, 28763648, 28763648, 28763648, 28763648, 41373184, 41373184, 41373184, 41373184])
    diskspace.append([16.8, 16.8, 16.8, 16.8, 110, 110, 110, 110, 158, 158, 158, 158])
    correlation.append(r)
    correlations['wmt15'].extend(r_15)
    correlations['wmt16'].extend(r_16)
    correlations['wmt21'].extend(r_21)

    print("Average WMT15 correlation: ", round(mean(correlations['wmt15']), 4))
    print("Average WMT16 correlation: ", round(mean(correlations['wmt16']), 4))
    print("Average WMT21 correlation: ", round(mean(correlations['wmt21']), 4))
    print("===========================================================")

    if method == 1:
        runtime_cpu = [j for i in runtime_cpu for j in i]
        runtime_gpu = [j for i in runtime_gpu for j in i]
        memory_gpu = [j for i in memory_gpu for j in i]
        params = [j for i in params for j in i]
        diskspace = [j for i in diskspace for j in i]
        correlation = [j for i in correlation for j in i]
        data = {'runtime_cpu': runtime_cpu,
                'runtime_gpu': runtime_gpu,
                'memory_gpu': memory_gpu,
                'params': params,
                'diskspace': diskspace,
                'correlation': correlation}
        df = pd.DataFrame(data)
        print(df)
        print(df.corr(method='pearson').round(4).to_string())
    else:   # method 2
        dfs = []
        # iterate over metrics to build separate correlation matrices
        for t_cpu, t_gpu, m_gpu, par, ds, r in zip(runtime_cpu, runtime_gpu, memory_gpu, params, diskspace, correlation):
            data = {'runtime_cpu': t_cpu,
                    'runtime_gpu': t_gpu,
                    'memory_gpu': m_gpu,
                    'params': par,
                    'diskspace': ds,
                    'correlation': r}
            df = pd.DataFrame(data)
            df_corr = df.corr(method='pearson')
            #print(df_corr.round(4).to_string())    # uncomment here to view separate df
            # invert negative correlations
            for i in range(6):
                for j in range(6):
                    if df_corr.iloc[i,j] < 0:
                        df_corr.iloc[i,j] = -df_corr.iloc[i,j]  # nested loops will invert every cell if negative
            dfs.append(df_corr)
        df_avg = pd.concat(dfs).groupby(level=0).mean()
        print(df_avg.round(4).to_string())

def load_avg_results(metric, models, distance=''):
    results_cpu = [load_results('envavg', 'cpu', metric, model, distance=distance) for model in models]
    runtime_cpu = [result[0][3] for result in results_cpu]
    results_gpu = [load_results('env3', 'gpu', metric, model, distance=distance) for model in models]
    runtime_gpu = [result[0][3] for result in results_gpu]
    memory_gpu = [result[1][3] for result in results_gpu]
    correlation = [(i[2][3]+j[2][3])/2 for i,j in zip(results_cpu, results_gpu)]
    r_15 = [(i[2][0]+j[2][0])/2 for i,j in zip(results_cpu, results_gpu)]
    r_16 = [(i[2][1]+j[2][1])/2 for i,j in zip(results_cpu, results_gpu)]
    r_21 = [(i[2][2]+j[2][2])/2 for i,j in zip(results_cpu, results_gpu)]
    return runtime_cpu, runtime_gpu, memory_gpu, correlation, r_15, r_16, r_21

if __name__ == "__main__":
    main(method)