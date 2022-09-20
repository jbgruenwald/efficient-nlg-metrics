import os

env = "envavg"
dev = "cgpu"
metric = "sentsim"
distance = ""
#models = ['bert', 'bert-tiny', 'distilbert', 'tinybert', 'deebert-mnli']
#models = ['bart-large-cnn+pth', 'bart-large-cnn', 'bart', 'distilbart66', 'distilbart123', 'distilbart-t2s']
#models = ['xlmr', 'mbert', 'distilmbert', 'xtremedistil', 'mminilm6', 'mminilm12']
#models = ['fs-tiny-b', 'fs-tiny-r', 'fs-tiny-d', 'fs-tiny-ms',
#          'fs-small-b', 'fs-small-r', 'fs-small-d', 'fs-small-ms']
#          'fs-medium-b', 'fs-medium-r', 'fs-medium-d', 'fs-medium-ms']
models = ['xlmr', 'mbert', 'distilmbert', 'xtremedistil', 'mminilm6', 'mminilm12', 'xlmr-2m-use2', 'xlmr-2m-pminilm12', 'xlmr-2m-pmpnet', 'mminilm12-2m-pminilm12']
#models = ['xlmr-refbased-5', 'xlmr-refbased-10', 'mminilm12-refbased-5', 'mminilm12-refbased-10']
runtime_chart = False


# configure path for results:
dir = "../../results/efficient-transformer-metrics/"

table_string = ""
chart_strings = {
    "wmt15": "",
    "wmt16": "",
    "wmt21": "",
    "wmtavg": ""
}
chart_cpu_runtime = chart_strings.copy()
chart_gpu_runtime = chart_strings.copy()
chart_memory = chart_strings.copy()
colors = {
    "bertscore": "blue",
    "moverscore": "green",
    "baryscore": "olive",
    "bartscore": "orange",
    "xmoverscore": "red",
    "sentsim": "cyan",
    "comet": "magenta",
    "transquest": "violet",
    "combinationscore": "teal",
    "frugalscore": "gray"
}

def prepare_table_and_chart(env, dev, metric, model, runtime_chart):
    cpu_runtime, _, cpu_correlation = load_results(env, "cpu", metric, model)
    if dev != 'cpu':
        gpu_runtime, memory, gpu_correlation = load_results("env3", "gpu", metric, model)
    else:
        gpu_runtime, memory, gpu_correlation = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]

    cpu_runtime_table = [format_duration(x) for x in cpu_runtime]
    gpu_runtime_table = [format_duration(x) for x in gpu_runtime]
    memory_table = [format_memvalue(x) for x in memory]
    correlation_table = [format_correlation((x+y)/2) for x, y in zip(cpu_correlation, gpu_correlation)]

    cpu_runtime_chart = [format_duration(x, True) for x in cpu_runtime]
    gpu_runtime_chart = [format_duration(x, True) for x in gpu_runtime]
    memory_chart = [format_memvalue(x, True) for x in memory]
    correlation_chart = [format_correlation((x+y)/2, True) for x, y in zip(cpu_correlation, gpu_correlation)]

    global table_string, chart_strings
    table_string += ('\multirow{4}{*}{%s}\n' if dev == 'cgpu' else '\multirow{2}{*}{%s}\n') % model
    table_string += '& CPU runtime & %s & %s & %s & %s &\n' % tuple(cpu_runtime_table)
    table_string += '& GPU runtime & %s & %s & %s & %s &\n' % tuple(gpu_runtime_table)
    if dev in ('gpu', 'cgpu'):
        table_string += '& memory      & %s & %s & %s & %s &\n' % tuple(memory_table)
    table_string += '& Pearson\'s r & %s & %s & %s & %s &\n' % tuple(correlation_table)

    chart_cpu_runtime["wmt15"] += '\chartscat{%s,%s}{%s}{%s}\n' % (cpu_runtime_chart[0], correlation_chart[0], colors[metric], model)
    chart_cpu_runtime["wmt16"] += '\chartscat{%s,%s}{%s}{%s}\n' % (cpu_runtime_chart[1], correlation_chart[1], colors[metric], model)
    chart_cpu_runtime["wmt21"] += '\chartscat{%s,%s}{%s}{%s}\n' % (cpu_runtime_chart[2], correlation_chart[2], colors[metric], model)
    chart_cpu_runtime["wmtavg"] += '\chartscat{%s,%s}{%s}{%s}\n' % (cpu_runtime_chart[3], correlation_chart[3], colors[metric], model)

    chart_gpu_runtime["wmt15"] += '\chartscat{%s,%s}{%s}{%s}\n' % (gpu_runtime_chart[0], correlation_chart[0], colors[metric], model)
    chart_gpu_runtime["wmt16"] += '\chartscat{%s,%s}{%s}{%s}\n' % (gpu_runtime_chart[1], correlation_chart[1], colors[metric], model)
    chart_gpu_runtime["wmt21"] += '\chartscat{%s,%s}{%s}{%s}\n' % (gpu_runtime_chart[2], correlation_chart[2], colors[metric], model)
    chart_gpu_runtime["wmtavg"] += '\chartscat{%s,%s}{%s}{%s}\n' % (gpu_runtime_chart[3], correlation_chart[3], colors[metric], model)

    chart_memory["wmt15"] += '\chartscat{%s,%s}{%s}{%s}\n' % (memory_chart[0], correlation_chart[0], colors[metric], model)
    chart_memory["wmt16"] += '\chartscat{%s,%s}{%s}{%s}\n' % (memory_chart[1], correlation_chart[1], colors[metric], model)
    chart_memory["wmt21"] += '\chartscat{%s,%s}{%s}{%s}\n' % (memory_chart[2], correlation_chart[2], colors[metric], model)
    chart_memory["wmtavg"] += '\chartscat{%s,%s}{%s}{%s}\n' % (memory_chart[3], correlation_chart[3], colors[metric], model)

def load_results(env, dev, metric, model, distance=distance):
    if env == "envavg":
        runtime1, memory1, correlation1 = load_results('env1', dev, metric, model, distance=distance)
        runtime3, memory3, correlation3 = load_results('env3', dev, metric, model, distance=distance)
        runtime = [ (x1+x2)/2 for x1, x2 in zip(runtime1, runtime3) ]
        memory = [ (x1+x2)/2 for x1, x2 in zip(memory1, memory3) ]
        correlation = [ (x1+x2)/2 for x1, x2 in zip(correlation1, correlation3) ]
        return runtime, memory, correlation
    else:

        avg_durations = []
        avg_memvalues = []
        avg_correlations = []

        for dataset in ('wmt15', 'wmt16', 'wmt21'):
            file = os.path.join(dir, env + '-' + dev, metric, '%s-%s-%s-%s-%s.txt' % (metric+distance, model, dataset, env, dev))

            if dataset == 'wmt15':
                sent = 2000
            elif dataset == 'wmt16':
                sent = 3360
            elif dataset == 'wmt20':
                sent = 4000
            else:  # wmt21
                sent = 3250
            with open(file) as f:
                values = [float(line.strip().split()[2]) for line in f]  # split the lines by spaces, take the third token of every line and create a list
            if len(values) < 9:
                print("Not enough measurements for %s on %s" % (model, dataset))
                exit()
            durations = values[::3]         # split the list
            memvalues = values[1::3]        # split the list
            correlations = values[2::3]     # split the list
            avg_durations.append(sum(durations) / len(durations) / sent)    # calculate average, normalize it
            avg_memvalues.append(sum(memvalues) / len(memvalues))           # calculate average
            avg_correlations.append(sum(correlations) / len(correlations))  # calculate average

        avg_durations.append(sum(avg_durations) / len(avg_durations))
        avg_memvalues.append(sum(avg_memvalues) / len(avg_memvalues))
        avg_correlations.append(sum(avg_correlations) / len(avg_correlations))

        return avg_durations, avg_memvalues, avg_correlations

def format_duration(x, for_chart=False):
    if for_chart:
        return str(round_formatted(x*1000)).rjust(4, ' ')
    else:
        return (f'{round_formatted(x*1000):,}ms').ljust(7, ' ')

def format_memvalue(x, for_chart=False):
    if for_chart:
        return str(round_formatted(x/1000000)).rjust(4, ' ')
    else:
        return (f'{round_formatted(x/1000000):,}MB').ljust(7, ' ')

def format_correlation(x, for_chart=False):
    if for_chart:
        return (f'{round(x, 4):,}').ljust(6, '0')
    else:
        return (f'{round(x, 4):,}').ljust(7, ' ')

def round_formatted(x):
    return round(x, 1) if x < 100 else round(x)

if __name__ == "__main__":
    for model in models:
        prepare_table_and_chart(env, dev, metric, model, runtime_chart)
    print(table_string)
    print('CPU runtime chart:\n' + chart_cpu_runtime["wmtavg"])
    print('GPU runtime chart:\n' + chart_gpu_runtime["wmtavg"])
    print('memory chart:\n' + chart_memory["wmtavg"])