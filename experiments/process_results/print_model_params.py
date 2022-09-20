from transformers import AutoModel
from experiments.config import model_name

params = []
for name in ('bart-large-cnn', 'bart', 'distilbart66', 'distilbart123', 'distilbart-t2s'):
    model = AutoModel.from_pretrained(model_name[name])
    params.append(sum([p.numel() for p in model.parameters() if p.requires_grad]))
print(params)