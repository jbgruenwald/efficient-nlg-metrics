from metrics.frugalscore.run_frugalscore import main, ModelArguments, DataTrainingArguments
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict
from experiments.config import model_name, frugalscore_cache_dir


def frugalscore(srcs, hyps, refs, model, model2, distance, langs, batch_size, device):
    model_args = ModelArguments(model_name_or_path=model_name[model], cache_dir=frugalscore_cache_dir)
    data_args = DataTrainingArguments(max_seq_length = 512, train_file='examplefile.json', validation_file='examplefile.json', test_file='examplefile.json')
    training_args = TrainingArguments(overwrite_output_dir=True, do_predict=True, output_dir='predictions', no_cuda=('cuda' not in device), per_device_eval_batch_size=1)

    data = DatasetDict({'test': Dataset.from_dict({'sentence1': refs, 'sentence2': hyps})})

    score = main(data, model_args, data_args, training_args)
    print(score)
    return {'score': score}