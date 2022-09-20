from metrics.bertscore.score import score
from experiments.config import model_name

def bertscore(srcs, hyps, refs, model, model2, distance, langs, batch_size, device):
    P, R, F1 = score(hyps, refs, lang="en", verbose=True, model_type=model_name[model], batch_size=batch_size, device=device)
    print(F1)
    return {'score': F1}