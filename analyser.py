import torch, json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_model     = None
_tokenizer = None
_config    = None
_device    = None

def load_model(save_path='saved_model/'):
    global _model, _tokenizer, _config, _device
    if _model is not None:
        return  # already loaded

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(save_path + 'config.json') as f:
        _config = json.load(f)

    _tokenizer = AutoTokenizer.from_pretrained(save_path + 'tokenizer/')
    _model     = AutoModelForSequenceClassification.from_pretrained(save_path + 'model/')
    _model.to(_device)
    _model.eval()


def analyse(text: str, threshold: float = None) -> dict:
    load_model()
    thresh = threshold if threshold is not None else _config['threshold']

    enc = _tokenizer(
        str(text), max_length=_config['max_len'],
        padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        logits = _model(
            input_ids=enc['input_ids'].to(_device),
            attention_mask=enc['attention_mask'].to(_device)
        ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    toxic = float(probs[1])
    cats  = _get_categories(toxic)

    return {
        'is_safe':     toxic < thresh,
        'label':       'SAFE' if toxic < thresh else 'TOXIC',
        'toxic_score': round(toxic, 4),
        'safe_score':  round(float(probs[0]), 4),
        'confidence':  round(float(max(probs)) * 100, 1),
        'categories':  cats,
        'meter':       int(toxic * 100),
    }


def _get_categories(score: float) -> list:
    """Score ke basis pe likely categories batao"""
    if score < 0.3:
        return ['✅ Clean content']
    elif score < 0.5:
        return ['⚠️ Mildly inappropriate']
    elif score < 0.7:
        return ['🔞 Possibly toxic', '⚠️ Review recommended']
    else:
        return ['🔴 Highly toxic', '🚫 Block recommended']


def get_model_info() -> dict:
    load_model()
    return {
        'model':    _config.get('model_name', 'DistilBERT'),
        'roc_auc':  _config.get('val_roc_auc', 'N/A'),
        'accuracy': _config.get('val_accuracy', 'N/A'),
        'device':   str(_device),
    }